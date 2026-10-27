#ifndef NOCDAS_NIPACKETASSEMBLER_HPP_
#define NOCDAS_NIPACKETASSEMBLER_HPP_
class NI;
class Packet;
// Legacy batch packetization/assembly; finite streaming buffers are a later model change.
class NIPacketAssembler {
public:
    static void poll(NI& ni);
};
#endif
