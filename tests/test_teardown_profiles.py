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
Acceptance tests for the session-scoped teardown & drain profile stacks.

The tests are deliberately split *by teardown exit*: half close on
``Channel.shutdown``, linger on ``Channel.close``, transport-level teardown on
``Transport.close``, SFTP handle drain on ``SFTPFile._close`` and agent
handling on ``SSHClient.close``.  Every exit must show its own measurably
distinct behavior -- a single switch consulted by one line in all five exits
would fail these tests.

Stack mechanics and per-session isolation (no cross-talk between concurrent
sessions pushing different profiles) are covered separately.
"""

import gc
import os
import struct
import sys
import threading
import time
import unittest

from paramiko import (
    Transport,
    SSHClient,
    AutoAddPolicy,
    RSAKey,
    SFTPClient,
    SFTPServer,
    TeardownProfile,
    TeardownPolicyError,
    HalfCloseMode,
    ChannelCloseLinger,
    TransportTeardown,
    SftpHandleFlush,
    BASELINE_PROFILE,
    teardown as teardown_module,
)
from paramiko.common import MSG_CHANNEL_EOF, MSG_CHANNEL_CLOSE
from paramiko.message import Message
from paramiko.channel import Channel
from paramiko.sftp import CMD_CLOSE

from tests.loop import LoopSocket
from tests.test_transport import NullServer
from tests.stub_sftp import StubServer, StubSFTPServer
from tests.util import test_path


HOST_KEY = RSAKey.from_private_key_file(test_path('test_rsa.key'))
PUBLIC_HOST_KEY = RSAKey(data=HOST_KEY.asbytes())

FEIGN = HalfCloseMode.FEIGN_READ_LOCAL
SEND_EOF = HalfCloseMode.SEND_EOF_WRITE
FORBID = HalfCloseMode.FORBID_HALF
LINGER = ChannelCloseLinger.LINGER_FOR_STATUS
UNLINK = ChannelCloseLinger.UNLINK_IMMEDIATE
CHANNELS_ONLY = TransportTeardown.CHANNELS_ONLY
STOP_UNLINK_SOCK = TransportTeardown.STOP_UNLINK_SOCK
CLIENT_WITH_AGENT = TransportTeardown.CLIENT_WITH_AGENT
SYNC_FINISH = SftpHandleFlush.SYNC_FINISH
ASYNC_NOWAIT = SftpHandleFlush.ASYNC_NOWAIT
REJECT_OPEN_HANDLES = SftpHandleFlush.REJECT_OPEN_HANDLES


class LoopbackMixin (object):
    def loopback(self, server=None):
        self.socks = LoopSocket()
        self.sockc = LoopSocket()
        self.sockc.link(self.socks)
        self.tc = Transport(self.sockc)
        self.ts = Transport(self.socks)
        self.ts.add_server_key(HOST_KEY)
        self.event = threading.Event()
        self.server = server if server is not None else NullServer()
        self.ts.start_server(self.event, self.server)
        self.tc.connect(hostkey=PUBLIC_HOST_KEY,
                        username='slowdive', password='pygmalion')
        self.assertTrue(self.event.wait(5.0))
        self.assertTrue(self.ts.is_active())

    def open_channel_pair(self):
        # open_session() itself blocks until OPEN_CONFIRMATION comes back
        chan = self.tc.open_session()
        schan = self.ts.accept(5.0)
        self.assertIsNotNone(schan)
        return chan, schan

    def cleanup_loopback(self):
        for t in (getattr(self, 'tc', None), getattr(self, 'ts', None)):
            if t is not None:
                try:
                    if t.is_active():
                        t.close()
                except Exception:
                    pass
        for s in (getattr(self, 'sockc', None), getattr(self, 'socks', None)):
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass


class MessageRecorder (object):
    """
    Records user-message codes per channel id by wrapping
    ``Transport._send_user_message``.
    """

    def __init__(self, transport):
        self.transport = transport
        self._orig = transport._send_user_message
        self.by_channel = {}
        self.codes = []
        transport._send_user_message = self._record

    def _record(self, msg):
        raw = msg.asbytes()
        code = raw[0]
        chanid = struct.unpack('>I', raw[1:5])[0] if len(raw) >= 5 else -1
        self.codes.append(code)
        self.by_channel.setdefault(chanid, []).append(code)
        return self._orig(msg)

    def for_channel(self, chan):
        return list(self.by_channel.get(chan.get_id(), []))

    def restore(self):
        self.transport._send_user_message = self._orig


class ChannelHalfCloseProfileTest (LoopbackMixin, unittest.TestCase):
    """Exit #1: Channel.shutdown / shutdown_read / shutdown_write."""

    def setUp(self):
        self.loopback()
        self.recorder = MessageRecorder(self.tc)

    def tearDown(self):
        self.recorder.restore()
        self.cleanup_loopback()

    # -- unframed behavior must stay identical to legacy -----------------

    def test_unframed_read_half_close_feigns_locally_without_eof_packet(self):
        chan, schan = self.open_channel_pair()
        chan.shutdown(0)
        self.assertEqual(1, chan.eof_received)
        self.assertEqual([], self.recorder.for_channel(chan))
        # and shutdown_read is the same exit
        chan2, schan2 = self.open_channel_pair()
        chan2.shutdown_read()
        self.assertEqual([], self.recorder.for_channel(chan2))

    def test_unframed_write_half_close_sends_channel_eof(self):
        chan, schan = self.open_channel_pair()
        chan.shutdown(1)
        self.assertEqual([MSG_CHANNEL_EOF], self.recorder.for_channel(chan))
        chan2, schan2 = self.open_channel_pair()
        chan2.shutdown_write()
        self.assertEqual([MSG_CHANNEL_EOF],
                         self.recorder.for_channel(chan2))

    def test_unframed_full_close_sends_eof_then_close(self):
        chan, schan = self.open_channel_pair()
        chan.close()
        self.assertEqual([MSG_CHANNEL_EOF, MSG_CHANNEL_CLOSE],
                         self.recorder.for_channel(chan))

    # -- feign_read_local frame -------------------------------------------

    def test_feign_read_local_read_half_emits_nothing(self):
        with self.tc.teardown_profiles().push(half_close_mode=FEIGN):
            chan, schan = self.open_channel_pair()
            chan.shutdown(0)
            self.assertEqual(1, chan.eof_received)
            self.assertEqual([], self.recorder.for_channel(chan))

    def test_feign_read_local_refuses_write_half_but_full_close_still_wires(self):
        with self.tc.teardown_profiles().push(half_close_mode=FEIGN):
            chan, schan = self.open_channel_pair()
            self.assertRaises(TeardownPolicyError, chan.shutdown, 1)
            self.assertRaises(TeardownPolicyError, chan.shutdown_write)
            # how=2 must be refused *before* any local half-action leaks
            self.assertRaises(TeardownPolicyError, chan.shutdown, 2)
            self.assertEqual(0, chan.eof_received)
            self.assertEqual([], self.recorder.for_channel(chan))
            # the full close profile is unaffected and still reaches the wire
            chan.close()
            self.assertEqual([MSG_CHANNEL_EOF, MSG_CHANNEL_CLOSE],
                             self.recorder.for_channel(chan))

    def test_feign_shutdown_vs_close_sequences_diverge(self):
        with self.tc.teardown_profiles().push(half_close_mode=FEIGN):
            chan, schan = self.open_channel_pair()
            chan.shutdown(0)
            self.assertEqual([], self.recorder.for_channel(chan))
            other, oschan = self.open_channel_pair()
            other.close()
            self.assertEqual([MSG_CHANNEL_EOF, MSG_CHANNEL_CLOSE],
                             self.recorder.for_channel(other))

    # -- send_eof_write frame ---------------------------------------------

    def test_send_eof_write_frame_enforces_wire_eof(self):
        with self.tc.teardown_profiles().push(half_close_mode=SEND_EOF):
            chan, schan = self.open_channel_pair()
            chan.shutdown(0)
            self.assertEqual([], self.recorder.for_channel(chan))
            chan2, schan2 = self.open_channel_pair()
            chan2.shutdown(1)
            self.assertEqual([MSG_CHANNEL_EOF],
                             self.recorder.for_channel(chan2))

    # -- forbid_half frame -------------------------------------------------

    def test_forbid_half_refuses_every_half_close(self):
        with self.tc.teardown_profiles().push(half_close_mode=FORBID):
            chan, schan = self.open_channel_pair()
            self.assertRaises(TeardownPolicyError, chan.shutdown, 0)
            self.assertRaises(TeardownPolicyError, chan.shutdown_read)
            self.assertRaises(TeardownPolicyError, chan.shutdown, 1)
            self.assertRaises(TeardownPolicyError, chan.shutdown_write)
            self.assertRaises(TeardownPolicyError, chan.shutdown, 2)
            self.assertEqual([], self.recorder.for_channel(chan))
            # only the full close profile is available
            chan.close()
            self.assertEqual([MSG_CHANNEL_EOF, MSG_CHANNEL_CLOSE],
                             self.recorder.for_channel(chan))

    # -- stack lifecycle ---------------------------------------------------

    def test_context_frame_auto_pops_even_on_exception(self):
        stack = self.tc.teardown_profiles()
        try:
            with stack.push(half_close_mode=FORBID):
                self.assertEqual(1, len(stack))
                raise RuntimeError('boom from user code')
        except RuntimeError:
            pass
        self.assertEqual(0, len(stack))
        # legacy behavior is observable again
        chan, schan = self.open_channel_pair()
        chan.shutdown(1)
        self.assertEqual([MSG_CHANNEL_EOF], self.recorder.for_channel(chan))

    def test_nested_frames_restore_outer_user_profile(self):
        stack = self.tc.teardown_profiles()
        with stack.push(half_close_mode=FORBID):
            chan, schan = self.open_channel_pair()
            self.assertRaises(TeardownPolicyError, chan.shutdown, 0)
            with stack.push(half_close_mode=FEIGN):
                # inner relaxation: read feign allowed again
                chan2, schan2 = self.open_channel_pair()
                chan2.shutdown(0)
                self.assertEqual([], self.recorder.for_channel(chan2))
            # back to the *outer* frame -- never to a process global
            chan3, schan3 = self.open_channel_pair()
            self.assertRaises(TeardownPolicyError, chan3.shutdown, 0)
        self.assertEqual(0, len(stack))
        chan4, schan4 = self.open_channel_pair()
        chan4.shutdown(0)
        self.assertEqual([], self.recorder.for_channel(chan4))

    def test_partial_frame_inherits_unset_axes_from_baseline(self):
        with self.tc.teardown_profiles().push(half_close_mode=FORBID):
            eff = self.tc.teardown_profiles().effective()
            self.assertEqual(FORBID, eff.half_close_mode)
            self.assertEqual(LINGER, eff.channel_close_linger)
            self.assertEqual(STOP_UNLINK_SOCK, eff.transport_teardown)
            self.assertEqual(SYNC_FINISH, eff.sftp_handle_flush)


class ChannelCloseLingerTest (LoopbackMixin, unittest.TestCase):
    """Exit #2: Channel.close linger/unlink disposition."""

    def setUp(self):
        self.loopback()

    def tearDown(self):
        self.cleanup_loopback()

    def _dispatch_exit_status_like_run_loop(self, chanid, code):
        """
        Reproduce exactly what Transport.run does with a post-close
        exit-status request (transport.py channel dispatch): look the
        channel up in the ChannelMap, deliver to Channel._handle_request
        when still linked, otherwise treat it as a dead-channel message.
        """
        m = Message()
        m.add_string('exit-status')
        m.add_boolean(False)
        m.add_int(code)
        # the run loop feeds freshly parsed Messages at stream position 0
        m.rewind()
        target = self.tc._channels.get(chanid)
        if target is not None:
            Channel._handle_request(target, m)

    def _assert_exit_status_after_close(self, frame_value, expected_status,
                                        should_unlink):
        chan, schan = self.open_channel_pair()
        chan.exec_command(b'yes')
        # Suppress the server's automatic CHANNEL_CLOSE reply: the run loop
        # dispatches through Transport._channel_handler_table, whose legacy
        # Channel._handle_close unlinks unconditionally and would race the
        # post-close meta-data delivery this exit is meant to govern.  An
        # instance-level table shadows the class table for the server side.
        table = dict(type(self.ts)._channel_handler_table)
        table[MSG_CHANNEL_CLOSE] = lambda ch, m: None
        self.ts._channel_handler_table = table
        try:
            stack = self.tc.teardown_profiles()
            with stack.push(channel_close_linger=frame_value):
                chan.close()
            # let the server run loop consume the CLOSE while its reply is
            # still suppressed, so no late reply can arrive after restore
            time.sleep(0.1)
            chanid = chan.get_id()
            if should_unlink:
                self.assertIsNone(self.tc._channels.get(chanid))
            else:
                self.assertIsNotNone(self.tc._channels.get(chanid))
            # metadata arriving after the local close: linger keeps the
            # channel dispatchable, unlink_immediate makes the run loop
            # drop the message
            self._dispatch_exit_status_like_run_loop(chanid, expected_status)
            delivered = -1 if should_unlink else expected_status
            self.assertEqual(delivered, chan.exit_status)
            self.assertEqual(delivered, chan.recv_exit_status())
        finally:
            del self.ts._channel_handler_table

    def test_unframed_close_lingers_for_exit_status(self):
        self._assert_exit_status_after_close(LINGER, 23, False)

    def test_linger_for_status_keeps_channel_linked(self):
        self._assert_exit_status_after_close(LINGER, 42, False)

    def test_unlink_immediate_detaches_and_drops_late_status(self):
        self._assert_exit_status_after_close(UNLINK, 7, True)


class TransportTeardownProfileTest (LoopbackMixin, unittest.TestCase):
    """Exit #3: Transport.close transport-level teardown."""

    def setUp(self):
        self.loopback()
        self.destroyed = False

    def tearDown(self):
        if not self.destroyed:
            self.cleanup_loopback()

    def test_unframed_close_stops_thread_unlinks_and_closes_socket(self):
        chan, schan = self.open_channel_pair()
        self.tc.close()
        self.assertFalse(self.tc.is_active())
        self.assertTrue(self.sockc._closed)
        self.destroyed = True

    def test_stop_unlink_sock_frame_matches_legacy_teardown(self):
        chan, schan = self.open_channel_pair()
        with self.tc.teardown_profiles().push(
                transport_teardown=STOP_UNLINK_SOCK):
            self.tc.close()
        self.assertFalse(self.tc.is_active())
        self.assertTrue(self.sockc._closed)
        self.assertIsNone(self.tc._channels.get(chan.get_id()))
        self.destroyed = True

    def test_channels_only_keeps_thread_socket_and_session_usable(self):
        chan1, schan1 = self.open_channel_pair()
        with self.tc.teardown_profiles().push(
                transport_teardown=CHANNELS_ONLY):
            self.tc.close()
            # channel-level close happened (EOF+CLOSE went out)
            self.assertTrue(chan1.closed)
        # ...but thread, socket and the session survive
        self.assertTrue(self.tc.is_active())
        self.assertFalse(self.sockc._closed)
        # the session is still usable: a fresh channel can be opened
        chan2 = self.tc.open_session()
        schan2 = self.ts.accept(5.0)
        self.assertIsNotNone(schan2)
        self.assertTrue(chan2.active)
        # now really tear down, outside the frame
        self.tc.close()
        self.assertFalse(self.tc.is_active())
        self.assertTrue(self.sockc._closed)
        self.destroyed = True


class FakeAgent (object):
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class SshClientTeardownProfileTest (LoopbackMixin, unittest.TestCase):
    """Exit #5: SSHClient.close (transport teardown + agent handling)."""

    def setUp(self):
        # raw server side of the loopback (client side is an SSHClient)
        self.socks = LoopSocket()
        self.sockc = LoopSocket()
        self.sockc.link(self.socks)
        self.ts = Transport(self.socks)
        self.ts.add_server_key(HOST_KEY)
        self.event = threading.Event()
        self.ts.start_server(self.event, NullServer())
        self.client = SSHClient()
        self.client.set_missing_host_key_policy(AutoAddPolicy())
        self.client.connect(hostname='', sock=self.sockc, username='slowdive',
                            password='pygmalion', look_for_keys=False,
                            allow_agent=False)
        self.assertTrue(self.event.wait(5.0))
        self.tc = self.client.get_transport()

    def tearDown(self):
        try:
            if self.client.get_transport() is not None and \
                    self.client.get_transport().is_active():
                self.client.close()
        except Exception:
            pass
        self.cleanup_loopback()

    def _attach_agent(self):
        agent = FakeAgent()
        self.client._agent = agent
        return agent

    def test_unframed_close_tears_down_transport_and_agent(self):
        agent = self._attach_agent()
        self.client.close()
        self.assertTrue(agent.closed)
        self.assertIsNone(self.client.get_transport())
        self.assertTrue(self.sockc._closed)
        self.destroyed = True

    def test_preconnect_frame_is_adopted_onto_the_transport(self):
        client = SSHClient()
        client.set_missing_host_key_policy(AutoAddPolicy())
        pre_stack = client.teardown_profiles()
        pre_stack.push(half_close_mode=FORBID)
        socks2 = LoopSocket()
        sockc2 = LoopSocket()
        sockc2.link(socks2)
        ts2 = Transport(socks2)
        ts2.add_server_key(HOST_KEY)
        ev = threading.Event()
        ts2.start_server(ev, NullServer())
        try:
            client.connect(hostname='', sock=sockc2, username='slowdive',
                           password='pygmalion', look_for_keys=False,
                           allow_agent=False)
            self.assertTrue(ev.wait(5.0))
            # same stack object now governs the transport side
            self.assertIs(pre_stack,
                          client.get_transport().teardown_profiles())
            chan = client.get_transport().open_session()
            self.assertTrue(chan.active)
            self.assertRaises(TeardownPolicyError, chan.shutdown, 0)
            pre_stack.pop()
        finally:
            client.close()
            ts2.close()
            socks2.close()
            sockc2.close()

    def test_client_with_agent_closes_both_transport_and_agent(self):
        agent = self._attach_agent()
        with self.client.teardown_profiles().push(
                transport_teardown=CLIENT_WITH_AGENT):
            self.client.close()
        self.assertTrue(agent.closed)
        self.assertIsNone(self.client._agent)
        self.assertTrue(self.sockc._closed)
        self.destroyed = True

    def test_stop_unlink_sock_frame_leaves_client_agent_alone(self):
        agent = self._attach_agent()
        with self.client.teardown_profiles().push(
                transport_teardown=STOP_UNLINK_SOCK):
            self.client.close()
        # transport fully torn down ...
        self.assertTrue(self.sockc._closed)
        self.assertIsNone(self.client.get_transport())
        # ... but this frame deliberately does not process the agent
        self.assertFalse(agent.closed)
        self.assertIs(agent, self.client._agent)
        self.destroyed = True

    def test_channels_only_frame_keeps_session_binding_and_agent(self):
        agent = self._attach_agent()
        with self.client.teardown_profiles().push(
                transport_teardown=CHANNELS_ONLY):
            self.client.close()
        # transport keeps running and stays bound to the client
        self.assertTrue(self.tc.is_active())
        self.assertIs(self.tc, self.client.get_transport())
        self.assertFalse(agent.closed)
        # session still usable from the client's transport
        chan = self.tc.open_session()
        self.assertTrue(chan.active)

    def test_bare_transport_close_cannot_impersonate_client_teardown(self):
        agent = self._attach_agent()
        # calling the transport exit directly must never touch the agent
        self.tc.close()
        self.assertFalse(agent.closed)
        self.assertIs(agent, self.client._agent)
        self.client._transport = None
        self.destroyed = True


def _start_sftp_loopback():
    socks = LoopSocket()
    sockc = LoopSocket()
    sockc.link(socks)
    tc = Transport(sockc)
    ts = Transport(socks)
    ts.add_server_key(HOST_KEY)
    event = threading.Event()
    ts.set_subsystem_handler('sftp', SFTPServer, StubSFTPServer)
    ts.start_server(event, StubServer())
    tc.connect(username='slowdive', password='pygmalion')
    event.wait(5.0)
    sftp = SFTPClient.from_transport(tc)
    return sftp, tc, ts, sockc, socks


class SftpHandleFlushProfileTest (unittest.TestCase):
    """Exit #4: SFTPFile.close / _close drain dispositions."""

    @classmethod
    def setUpClass(cls):
        cls.sftp, cls.tc, cls.ts, cls.sockc, cls.socks = \
            _start_sftp_loopback()
        cls.prefix = 'teardown_profile_%d_' % os.getpid()
        cls._made = []

    @classmethod
    def tearDownClass(cls):
        for name in list(cls._made):
            try:
                cls.sftp.remove(name)
            except Exception:
                pass
        for t, s in ((cls.tc, cls.sockc), (cls.ts, cls.socks)):
            try:
                t.close()
            except Exception:
                pass
            try:
                s.close()
            except Exception:
                pass

    def _path(self, name):
        p = self.prefix + name + '_%d' % id(self)
        self._made.append(p)
        return p

    def _install_spies(self):
        calls = {'finish': 0, 'sync_close': 0, 'async_close': 0}
        sftp = self.sftp
        orig_finish = sftp._finish_responses
        orig_request = sftp._request
        orig_async = sftp._async_request
        # _request() itself funnels through _async_request(); the marker
        # keeps a synchronous CMD_CLOSE from being counted twice
        in_sync = [False]

        def finish(f):
            calls['finish'] += 1
            return orig_finish(f)

        def request(t, *args):
            if t == CMD_CLOSE:
                calls['sync_close'] += 1
            in_sync[0] = True
            try:
                return orig_request(t, *args)
            finally:
                in_sync[0] = False

        def async_request(fileobj, t, *args):
            if t == CMD_CLOSE and not in_sync[0]:
                calls['async_close'] += 1
            return orig_async(fileobj, t, *args)

        sftp._finish_responses = finish
        sftp._request = request
        sftp._async_request = async_request
        self.addCleanup(self._restore_spies, sftp,
                        orig_finish, orig_request, orig_async)
        return calls

    @staticmethod
    def _restore_spies(sftp, finish, request, async_request):
        sftp._finish_responses = finish
        sftp._request = request
        sftp._async_request = async_request

    def _pipelined_file(self, name, chunks=12, chunk=4096):
        path = self._path(name)
        f = self.sftp.open(path, 'wb')
        f.set_pipelined(True)
        payload = b'x' * chunk
        for _ in range(chunks):
            f.write(payload)
        return f, path

    def test_unframed_pipelined_close_drains_and_waits_for_close_response(self):
        calls = self._install_spies()
        f, path = self._pipelined_file('plain')
        self.assertTrue(f.pipelined)
        f.close()
        self.assertEqual(1, calls['finish'])
        self.assertEqual(1, calls['sync_close'])
        self.assertEqual(0, calls['async_close'])
        # data really landed
        self.assertEqual(12 * 4096, self.sftp.stat(path).st_size)

    def test_sync_finish_frame_drains_then_synchronous_close(self):
        calls = self._install_spies()
        f, path = self._pipelined_file('sync')
        with self.sftp.sock.get_transport().teardown_profiles().push(
                sftp_handle_flush=SYNC_FINISH):
            f.close()
        self.assertEqual(1, calls['finish'])
        self.assertEqual(1, calls['sync_close'])
        self.assertEqual(0, calls['async_close'])
        self.assertEqual(12 * 4096, self.sftp.stat(path).st_size)

    def test_async_nowait_frame_fires_close_without_drain_or_wait(self):
        calls = self._install_spies()
        f, path = self._pipelined_file('async')
        with self.sftp.sock.get_transport().teardown_profiles().push(
                sftp_handle_flush=ASYNC_NOWAIT):
            f.close()
        # never drained, never blocked on a response
        self.assertEqual(0, calls['finish'])
        self.assertEqual(0, calls['sync_close'])
        self.assertEqual(1, calls['async_close'])
        self.assertTrue(f._closed)
        # pump the loopback so the server processes the fired close
        deadline = time.time() + 5
        size = None
        while time.time() < deadline:
            try:
                size = self.sftp.stat(path).st_size
                break
            except Exception:
                time.sleep(0.05)
        self.assertEqual(12 * 4096, size)

    def test_reject_open_handles_is_neither_sync_nor_async(self):
        calls = self._install_spies()
        p1 = self._path('rej1')
        p2 = self._path('rej2')
        f1 = self.sftp.open(p1, 'wb')
        f2 = self.sftp.open(p2, 'wb')
        f1.write(b'one')
        f2.write(b'two')
        stack = self.sftp.sock.get_transport().teardown_profiles()
        with stack.push(sftp_handle_flush=REJECT_OPEN_HANDLES):
            # explicit close while a sibling handle is open: refused,
            # nothing is sent and the handle stays open
            self.assertRaises(TeardownPolicyError, f1.close)
            self.assertFalse(f1._closed)
            self.assertEqual(0, calls['sync_close'])
            self.assertEqual(0, calls['async_close'])
            # implicit (GC-style) async teardown is refused as well
            self.assertRaises(TeardownPolicyError, f1._close, async_=True)
            self.assertFalse(f1._closed)
            # __del__ must surface no traceback but also not silently close
            f1.__del__()
            self.assertFalse(f1._closed)
            self.assertEqual(0, calls['sync_close'])
            self.assertEqual(0, calls['async_close'])
        # outside the frame the handles close normally
        f1.close()
        f2.close()
        self.assertEqual(2, calls['sync_close'])

    def test_reject_open_handles_allows_closing_the_last_handle(self):
        calls = self._install_spies()
        p = self._path('rejlast')
        f = self.sftp.open(p, 'wb')
        f.write(b'only')
        with self.sftp.sock.get_transport().teardown_profiles().push(
                sftp_handle_flush=REJECT_OPEN_HANDLES):
            f.close()
        self.assertEqual(1, calls['sync_close'])
        self.assertEqual(0, calls['async_close'])

    def test_reject_open_handles_fails_session_teardown_loudly(self):
        # dedicated session so the shared class session survives
        sftp, tc, ts, sockc, socks = _start_sftp_loopback()
        try:
            path = 'teardown_profile_gate_%d' % os.getpid()
            f = sftp.open(path, 'wb')
            f.write(b'data')
            stack = tc.teardown_profiles()
            with stack.push(sftp_handle_flush=REJECT_OPEN_HANDLES):
                self.assertRaises(TeardownPolicyError, tc.close)
            # gate fired before anything destructive; the with-block has
            # already removed the frame again
            self.assertEqual(0, len(stack))
            self.assertTrue(tc.is_active())
            self.assertFalse(sockc._closed)
            f.close()
            sftp.remove(path)
            self.assertEqual(0, len(stack.open_sftp_handles()))
        finally:
            tc.close()
            ts.close()
            sockc.close()
            socks.close()

    def test_unframed_double_close_does_not_reissue_cmd_close(self):
        calls = self._install_spies()
        f, path = self._pipelined_file('double', chunks=1, chunk=100)
        f.close()
        f.close()
        self.assertEqual(1, calls['sync_close'])


class TeardownStackMechanicsTest (unittest.TestCase):
    """Stack semantics: baselines, merging, pop discipline, isolation."""

    def setUp(self):
        # Transport(None) never starts a thread; its session stack is still
        # mounted normally.
        self.s1 = Transport(None)
        self.s2 = Transport(None)
        self.stacks = (self.s1.teardown_profiles(),
                       self.s2.teardown_profiles())

    def tearDown(self):
        for t in (self.s1, self.s2):
            t.close()

    def test_effective_starts_at_per_session_baseline(self):
        self.assertEqual(BASELINE_PROFILE, self.s1.teardown_profiles().effective())
        self.assertEqual(BASELINE_PROFILE, self.s2.teardown_profiles().effective())
        self.assertIsNot(self.s1.teardown_profiles(),
                         self.s2.teardown_profiles())

    def test_explicit_distinguishes_pushed_value_from_legacy_baseline(self):
        stack = self.s1.teardown_profiles()
        self.assertIsNone(stack.explicit('transport_teardown'))
        with stack.push(half_close_mode=FORBID):
            self.assertIsNone(stack.explicit('transport_teardown'))
            self.assertEqual(FORBID, stack.explicit('half_close_mode'))
            self.assertEqual(STOP_UNLINK_SOCK,
                             stack.effective().transport_teardown)

    def test_push_kwargs_and_profile_object_equivalent(self):
        stack = self.s1.teardown_profiles()
        with stack.push(TeardownProfile(half_close_mode=FEIGN)):
            self.assertEqual(FEIGN, stack.effective().half_close_mode)
        with stack.push(half_close_mode=FEIGN):
            self.assertEqual(FEIGN, stack.effective().half_close_mode)

    def test_invalid_axis_value_rejected(self):
        self.assertRaises(ValueError, TeardownProfile,
                          half_close_mode='nonsense')

    def test_pop_empty_and_out_of_order_pop_raise(self):
        stack = self.s1.teardown_profiles()
        self.assertRaises(TeardownPolicyError, stack.pop)
        outer = stack.push(half_close_mode=FORBID)
        inner = stack.push(channel_close_linger=UNLINK)
        # an outer token cannot pop the inner frame
        self.assertRaises(TeardownPolicyError, stack._pop_token, outer)
        stack._pop_token(inner)
        stack._pop_token(outer)
        self.assertEqual(0, len(stack))

    def test_registry_returns_same_stack_for_same_session_only(self):
        again = teardown_module.stack_for_session(self.s1)
        self.assertIs(self.s1.teardown_profiles(), again)
        self.assertIsNot(again,
                         teardown_module.stack_for_session(self.s2))

    def test_registry_does_not_keep_dead_sessions_alive(self):
        t = Transport(None)
        stack = t.teardown_profiles()
        self.assertIs(stack, teardown_module.stack_for_session(t))
        del t
        gc.collect()
        t2 = Transport(None)
        self.assertIsNot(stack, t2.teardown_profiles())

    def test_two_concurrent_sessions_never_cross_stacks(self):
        a = self.s1.teardown_profiles()
        b = self.s2.teardown_profiles()
        errors = []
        iterations = 50
        a_push = threading.Barrier(2)
        both_pushed = threading.Barrier(2)
        done = threading.Barrier(2)

        def session_a():
            try:
                for _ in range(iterations):
                    with a.push(half_close_mode=FORBID):
                        a_push.wait()
                        both_pushed.wait()
                        # own frame visible
                        if a.effective().half_close_mode != FORBID:
                            errors.append('A lost its own frame')
                        # the other session's frame must never leak here
                        if b.effective().half_close_mode == FORBID:
                            errors.append('B frame leaked into A check')
                        done.wait()
            except Exception as e:
                errors.append(('A', repr(e)))

        def session_b():
            try:
                for _ in range(iterations):
                    with b.push(half_close_mode=FEIGN):
                        a_push.wait()
                        both_pushed.wait()
                        if b.effective().half_close_mode != FEIGN:
                            errors.append('B lost its own frame')
                        if a.effective().half_close_mode == FEIGN:
                            errors.append('A frame leaked into B check')
                        done.wait()
            except Exception as e:
                errors.append(('B', repr(e)))

        ta = threading.Thread(target=session_a)
        tb = threading.Thread(target=session_b)
        ta.start()
        tb.start()
        ta.join(10)
        tb.join(10)
        self.assertFalse(ta.is_alive())
        self.assertFalse(tb.is_alive())
        self.assertEqual([], errors)
        self.assertEqual(0, len(a))
        self.assertEqual(0, len(b))
        self.assertEqual(BASELINE_PROFILE, a.effective())
        self.assertEqual(BASELINE_PROFILE, b.effective())


if __name__ == '__main__':
    unittest.main()
