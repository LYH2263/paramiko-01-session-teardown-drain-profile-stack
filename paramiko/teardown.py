# Copyright (C) 2026  Paramiko Engineers
#
# This file is part of Paramiko.
#
# Paramiko is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 2.1 of the License, or (at your option)
# any later version.
#
# Paramiko is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Paramiko; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA.

"""
Session-scoped teardown & drain profile stacks.

This module is deliberately independent of the channel flow-control window
machinery, host key ``MissingHostKeyPolicy`` and the ``auth_handler``
authentication method chain.  It only governs how the five distinct teardown
exits behave within the lifetime of one SSH session (an ``SSHClient`` and/or
its underlying ``Transport``):

* ``Channel.shutdown`` / ``shutdown_read`` / ``shutdown_write`` -- half close,
* ``Channel.close`` -- full channel close and whether it stays linked to the
  ``Transport`` while post-close meta-data (e.g. ``exit-status``) may arrive,
* ``Transport.close`` -- transport level teardown,
* ``SFTPFile._close`` / ``SFTPFile.close`` -- SFTP handle drain/close,
* ``SSHClient.close`` -- transport teardown plus closing the SSH agent.

A session owns a :class:`SessionTeardownStack` (mounted through the shared,
thread safe :func:`stack_for_session` registry).  Every thread participating in
the same session enters the *same* stack; two different sessions always have two
different stacks -- a frame pushed on one session can never be observed on
another session, even when the sessions are used concurrently.  When no frame
has been pushed, every teardown exit follows the legacy behavior byte-for-byte.

Frames are pushed/popped explicitly or used as context managers; the context
manager form pops its own frame on any exit path, including exceptions, and
rejects out-of-order pops so a frame can never bounce the stack back to some
process-global default.
"""

import threading
import weakref

from paramiko.ssh_exception import SSHException


class TeardownPolicyError(SSHException):
    """
    A teardown/half-close/drain exit was attempted in a shape the active
    session teardown profile forbids (for example, a half close while
    ``forbid_half`` is active, or a session teardown while SFTP handles are
    still open under ``reject_open_handles``).
    """
    pass


class HalfCloseMode (object):
    """
    Dispositions of ``Channel.shutdown`` / ``shutdown_read`` /
    ``shutdown_write`` inside a teardown frame.

    * ``FEIGN_READ_LOCAL`` -- the read-direction half close may only feign EOF
      locally (``Channel.eof_received``); the half-close path must not emit a
      ``CHANNEL_EOF`` packet.  A write-direction half close, whose only purpose
      is that packet, is refused -- callers must use the full ``Channel.close``
      profile.
    * ``SEND_EOF_WRITE`` -- the read half feigns EOF locally while the write
      half must send ``CHANNEL_EOF`` via ``Channel._send_eof``.  This is the
      legacy behavior.
    * ``FORBID_HALF`` -- all three half-close entry points are refused; the
      caller must use ``Channel.close``.
    """
    FEIGN_READ_LOCAL = 'feign_read_local'
    SEND_EOF_WRITE = 'send_eof_write'
    FORBID_HALF = 'forbid_half'

    _ALL = (FEIGN_READ_LOCAL, SEND_EOF_WRITE, FORBID_HALF)  # noqa: F821


class ChannelCloseLinger (object):
    """
    Dispositions of ``Channel.close`` with respect to the channel's link in
    the owning ``Transport``.

    * ``LINGER_FOR_STATUS`` -- send ``CHANNEL_EOF`` + ``CHANNEL_CLOSE`` but
      keep the channel registered so post-close meta-data such as
      ``exit-status`` can still be delivered.  This is the legacy behavior.
    * ``UNLINK_IMMEDIATE`` -- detach the channel from the ``Transport``
      immediately after sending the close sequence.  Later ``exit-status``
      messages are then treated as messages for a dead channel.
    """
    LINGER_FOR_STATUS = 'linger_for_status'
    UNLINK_IMMEDIATE = 'unlink_immediate'

    _ALL = (LINGER_FOR_STATUS, UNLINK_IMMEDIATE)  # noqa: F821


class TransportTeardown (object):
    """
    Dispositions of the transport-level teardown exit.

    * ``CHANNELS_ONLY`` -- only perform channel-level closes for the currently
      registered channels.  The transport thread is not stopped and the socket
      is not closed; the session stays usable.
    * ``STOP_UNLINK_SOCK`` -- the legacy ``Transport.close`` shape: stop the
      transport thread, unlink every registered channel and close the socket.
    * ``CLIENT_WITH_AGENT`` -- like ``STOP_UNLINK_SOCK`` for the transport
      itself, and additionally marks the ``SSHClient.close`` exit as one that
      must also close the SSH agent.  It must not be impersonated by only
      calling ``Transport.close`` (which knows nothing about the agent).
    """
    CHANNELS_ONLY = 'channels_only'
    STOP_UNLINK_SOCK = 'stop_unlink_sock'
    CLIENT_WITH_AGENT = 'client_with_agent'

    _ALL = (CHANNELS_ONLY, STOP_UNLINK_SOCK, CLIENT_WITH_AGENT)  # noqa: F821


class SftpHandleFlush (object):
    """
    Dispositions of open SFTP handles at close/teardown.

    * ``SYNC_FINISH`` -- pipelined responses are drained via
      ``SFTPClient._finish_responses`` and the ``CMD_CLOSE`` is sent
      synchronously (the response is waited for).  Legacy behavior of
      ``SFTPFile.close``.
    * ``ASYNC_NOWAIT`` -- never drain and never block waiting for the close
      response; the ``CMD_CLOSE`` is fired through ``_async_request``.
    * ``REJECT_OPEN_HANDLES`` -- implicit (fire-and-forget) handle teardown is
      refused, an explicit close while sibling handles are still open is
      refused, and a session teardown while handles remain open fails loudly
      instead of silently falling back to async close.
    """
    SYNC_FINISH = 'sync_finish'
    ASYNC_NOWAIT = 'async_nowait'
    REJECT_OPEN_HANDLES = 'reject_open_handles'

    _ALL = (SYNC_FINISH, ASYNC_NOWAIT, REJECT_OPEN_HANDLES)  # noqa: F821


_AXES = ('half_close_mode', 'channel_close_linger',
         'transport_teardown', 'sftp_handle_flush')


class TeardownProfile (object):
    """
    One immutable teardown frame.

    Every axis is optional; an unset axis (``None``) neither tightens nor
    loosens anything and is inherited from the frame below (or from the
    per-session baseline).  This lets nested middleware override a single
    concern -- e.g. only ``half_close_mode`` -- without restating the other
    axes.
    """

    __slots__ = _AXES

    def __init__(self, half_close_mode=None, channel_close_linger=None,
                 transport_teardown=None, sftp_handle_flush=None):
        if half_close_mode is not None and \
                half_close_mode not in HalfCloseMode._ALL:
            raise ValueError('invalid half_close_mode %r' % half_close_mode)
        if channel_close_linger is not None and \
                channel_close_linger not in ChannelCloseLinger._ALL:
            raise ValueError(
                'invalid channel_close_linger %r' % channel_close_linger)
        if transport_teardown is not None and \
                transport_teardown not in TransportTeardown._ALL:
            raise ValueError(
                'invalid transport_teardown %r' % transport_teardown)
        if sftp_handle_flush is not None and \
                sftp_handle_flush not in SftpHandleFlush._ALL:
            raise ValueError(
                'invalid sftp_handle_flush %r' % sftp_handle_flush)
        self.half_close_mode = half_close_mode
        self.channel_close_linger = channel_close_linger
        self.transport_teardown = transport_teardown
        self.sftp_handle_flush = sftp_handle_flush

    def _over(self, lower):
        """Return this profile merged on top of ``lower`` (per-axis override)."""
        return TeardownProfile(
            half_close_mode=(self.half_close_mode
                             if self.half_close_mode is not None
                             else lower.half_close_mode),
            channel_close_linger=(self.channel_close_linger
                                  if self.channel_close_linger is not None
                                  else lower.channel_close_linger),
            transport_teardown=(self.transport_teardown
                                if self.transport_teardown is not None
                                else lower.transport_teardown),
            sftp_handle_flush=(self.sftp_handle_flush
                               if self.sftp_handle_flush is not None
                               else lower.sftp_handle_flush),
        )

    def __eq__(self, other):
        return isinstance(other, TeardownProfile) and all(
            getattr(self, axis) == getattr(other, axis) for axis in _AXES)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(getattr(self, axis) for axis in _AXES))

    def __repr__(self):
        overridden = ', '.join(
            '%s=%r' % (axis, getattr(self, axis))
            for axis in _AXES if getattr(self, axis) is not None)
        return '<TeardownProfile %s>' % (overridden or 'inherit-all')


#: Session baseline: every axis reproduces the legacy teardown behavior.  Each
#: session gets this as its *own* starting point; it is never a mutable global
#: singleton that inner frames could pop back into.
BASELINE_PROFILE = TeardownProfile(
    half_close_mode=HalfCloseMode.SEND_EOF_WRITE,
    channel_close_linger=ChannelCloseLinger.LINGER_FOR_STATUS,
    transport_teardown=TransportTeardown.STOP_UNLINK_SOCK,
    sftp_handle_flush=SftpHandleFlush.SYNC_FINISH,
)


class _PushedProfile (object):
    """Push token; doubles as a context manager that pops exactly this frame."""

    def __init__(self, stack, profile):
        self._stack = stack
        self.profile = profile
        self._popped = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stack._pop_token(self)
        # never swallow the in-body exception; the frame is only removed
        return False

    def __repr__(self):
        return '<pushed %r on %r>' % (self.profile, self._stack)


class SessionTeardownStack (object):
    """
    Push/pop stack of :class:`TeardownProfile` frames for one session.

    The stack is safe to enter from any thread participating in the session
    (user threads and the transport thread).  Its effective view is the
    per-session baseline merged with the pushed frames bottom-up.
    """

    def __init__(self, session=None):
        # weak back-reference: the session owns this stack, so avoid a cycle
        self._session_ref = weakref.ref(session) if session is not None \
            else lambda: None
        self._lock = threading.RLock()
        self._frames = []
        # The baseline is per-session state (a copy, not the shared constant)
        # so an inner pop can only return to this session's outer profile.
        self._baseline = TeardownProfile(
            half_close_mode=BASELINE_PROFILE.half_close_mode,
            channel_close_linger=BASELINE_PROFILE.channel_close_linger,
            transport_teardown=BASELINE_PROFILE.transport_teardown,
            sftp_handle_flush=BASELINE_PROFILE.sftp_handle_flush,
        )
        # open SFTPFile handles reachable from this session (weak: tracking
        # must not extend handle lifetimes)
        self._open_sftp_handles = weakref.WeakSet()

    # -- stack operations ------------------------------------------------

    def push(self, profile=None, **axes):
        """
        Push a frame and return a token which is also a context manager.

        Either pass a ready :class:`TeardownProfile` or build one in place via
        keyword axes (``half_close_mode=...`` etc).
        """
        if profile is None:
            profile = TeardownProfile(**axes)
        elif axes:
            profile = TeardownProfile(
                half_close_mode=axes.get('half_close_mode',
                                         profile.half_close_mode),
                channel_close_linger=axes.get('channel_close_linger',
                                              profile.channel_close_linger),
                transport_teardown=axes.get('transport_teardown',
                                            profile.transport_teardown),
                sftp_handle_flush=axes.get('sftp_handle_flush',
                                           profile.sftp_handle_flush),
            )
        elif not isinstance(profile, TeardownProfile):
            raise TypeError('expected TeardownProfile, got %r' % (profile,))
        token = _PushedProfile(self, profile)
        with self._lock:
            self._frames.append(token)
        return token

    def pop(self):
        """
        Pop and return the top frame's profile.

        :raises TeardownPolicyError: when no frame is pushed.
        """
        with self._lock:
            if not self._frames:
                raise TeardownPolicyError(
                    'cannot pop teardown profile: stack is empty')
            token = self._frames.pop()
        token._popped = True
        return token.profile

    def _pop_token(self, token):
        with self._lock:
            if token._popped:
                return
            if not self._frames or self._frames[-1] is not token:
                raise TeardownPolicyError(
                    'teardown profile popped out of order; refusing to unwind '
                    'onto a foreign frame')
            self._frames.pop()
            token._popped = True

    def __len__(self):
        with self._lock:
            return len(self._frames)

    def __bool__(self):
        return self.__len__() > 0

    __nonzero__ = __bool__

    def has_frames(self):
        """Whether any user frame is currently pushed on this session."""
        with self._lock:
            return bool(self._frames)

    def effective(self):
        """
        Effective profile: the session baseline merged with all pushed frames
        from the outermost to the innermost.
        """
        with self._lock:
            result = self._baseline
            for token in self._frames:
                result = token.profile._over(result)
            return result

    def explicit(self, axis):
        """
        Return the topmost value explicitly set for ``axis`` by a pushed
        frame, or ``None`` if no pushed frame sets it.

        Unlike :meth:`effective`, this distinguishes "a frame deliberately
        selected the baseline-shaped value" from "no frame touches this axis"
        -- important at exits whose legacy behavior differs from the baseline
        constant (e.g. ``SSHClient.close`` also closes the SSH agent).
        """
        with self._lock:
            for token in reversed(self._frames):
                value = getattr(token.profile, axis)
                if value is not None:
                    return value
        return None

    # -- SFTP handle registry used by REJECT_OPEN_HANDLES ----------------

    def register_open_handle(self, sftp_file):
        with self._lock:
            self._open_sftp_handles.add(sftp_file)

    def unregister_open_handle(self, sftp_file):
        with self._lock:
            self._open_sftp_handles.discard(sftp_file)

    def open_sftp_handles(self):
        """Snapshot of currently open (non-closed) tracked SFTP handles."""
        with self._lock:
            return [f for f in list(self._open_sftp_handles)
                    if not getattr(f, '_closed', False)]

    def enforce_no_open_sftp_handles(self):
        """
        Raise :class:`TeardownPolicyError` if tracked SFTP handles are still
        open; otherwise return normally.
        """
        handles = self.open_sftp_handles()
        if handles:
            raise TeardownPolicyError(
                'refusing session teardown: %d SFTP handle(s) still open; '
                'close them explicitly (silent async close is forbidden by '
                'the active teardown profile)' % len(handles))


#: Shared mount point for per-session stacks.  All sessions pass through this
#: registry, but entries are keyed by the session object itself, so concurrent
#: sessions never share a frame stack.  Values are held strongly by the
#: session (``Transport._teardown_stack``); this map is weak and does not keep
#: closed sessions alive.
_SESSION_STACKS = weakref.WeakKeyDictionary()
_REGISTRY_LOCK = threading.Lock()


def stack_for_session(session, create=True):
    """
    Return the :class:`SessionTeardownStack` bound to ``session``.

    The same object is returned for every call with the same session; distinct
    sessions get distinct stacks.
    """
    with _REGISTRY_LOCK:
        stack = _SESSION_STACKS.get(session)
        if stack is None and create:
            stack = SessionTeardownStack(session)
            _SESSION_STACKS[session] = stack
        return stack


def bind_session_stack(session, stack):
    """
    (Re)bind an existing stack onto ``session``.

    Used when an ``SSHClient``'s pre-connect stack is adopted by the
    ``Transport`` created in ``SSHClient.connect`` so frames pushed before the
    connection survive into the session.
    """
    with _REGISTRY_LOCK:
        _SESSION_STACKS[session] = stack
    return stack
