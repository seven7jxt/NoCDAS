#ifndef NOCDAS_NIPACKETIZER_HPP_
#define NOCDAS_NIPACKETIZER_HPP_
class NI;
class Packet;
// Legacy batch packetization/assembly; finite streaming buffers are a later model change.
class NIPacketizer {
public:
    static bool packetize(NI& ni, Packet* packet, int vn);
};
#endif
