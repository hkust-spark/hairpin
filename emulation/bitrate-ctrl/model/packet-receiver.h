#ifndef PACKET_RECEIVER_H
#define PACKET_RECEIVER_H

#include "common-header.h"
#include "fec-policy.h"
#include "ns3/socket.h"
#include "ns3/simulator.h"
#include "ns3/nstime.h"
#include "ns3/timer.h"
#include <vector>
#include <queue>

namespace ns3 {

class GameClient;

class PacketReceiver : public Object {
public:
    static TypeId GetTypeId (void);
    PacketReceiver(GameClient *, void (GameClient::*)(Ptr<VideoPacket>), Ptr<Socket>, uint32_t);
    ~PacketReceiver();

    void OnSocketRecv_receiver(Ptr<Socket>);

    void Feedback_NetState();

    /**
     * Mainly for sending ACK packet
     */
    void SendPacket(Ptr<NetworkPacket>);

    void Set_FECgroup_delay(Time&);
    Time Get_FECgroup_delay();
    // API for setting FEC group delay @kongxiao0532

    void StopRunning();

    bool lessThan (uint16_t id1, uint16_t id2);

    bool lessThan_simple(uint16_t id1, uint16_t id2);

private:
    GameClient * game_client;
    Ptr<Socket> m_socket;
    void (GameClient::*RcvPacketFunc)(Ptr<VideoPacket>);

    std::deque<Ptr<RcvTime>> m_record;

    Time     m_feedback_interval;
    Timer    m_feedbackTimer;
    uint16_t m_last_feedback;

    /* Calculate statistics within a sliding window */
    uint16_t wnd_size;
    uint32_t time_wnd_size; // in us
    uint16_t m_credible;
    uint32_t bytes_per_packet;
    uint16_t last_id;

    uint16_t pkts_in_wnd;
    uint32_t bytes_in_wnd;
    uint32_t time_in_wnd;
    uint32_t losses_in_wnd;

    float loss_rate;
    float throughput_kbps;
    Time  one_way_dispersion;
    std::vector<int> m_loss_seq;
    std::vector<Ptr<RcvTime>> m_recv_sample;


};  // class PacketReceiver

};  // namespace ns3

#endif  /* PACKET_RECEIVER_H */