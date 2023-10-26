#ifndef PACKET_SENDER_H
#define PACKET_SENDER_H

#include "common-header.h"
#include "fec-policy.h"
#include "ns3/sender-based-controller.h"
#include "ns3/gcc-controller.h"
#include "ns3/nada-controller.h"

#include "ns3/core-module.h"
#include "ns3/socket.h"
#include "ns3/application.h"
#include "ns3/simulator.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/nstime.h"
#include "ns3/timer.h"
#include <vector>
#include <queue>
#include <memory>

namespace ns3 {

class GameServer;

class SentPacketInfo : public Object {
public:
    SentPacketInfo(uint16_t,uint16_t,Time,PacketType,bool,uint16_t);
    ~SentPacketInfo();
    uint16_t pkt_id;
    uint16_t batch_id;
    Time pkt_send_time;
    Time pkt_ack_time;
    PacketType pkt_type;
    bool is_goodput;
    uint16_t pkt_size; //in bytes
}; //class SentPacketInfo

class PacketFrame : public Object{
public:
    static TypeId GetTypeId (void);
    PacketFrame(std::vector<Ptr<VideoPacket>> packets, bool retransmission);
    ~PacketFrame();

    std::vector<Ptr<VideoPacket>> packets_in_Frame;

    Time Frame_encode_time_; /*the encodde time of the packets in this frame */

    uint32_t per_packet_size; /* bytes in each packet */

    bool retransmission; /* whether the packets are retransmission packets */

    /**
     * \brief calculate the total bytes in this frame
     */
    uint32_t Frame_size_in_byte();

    /**
     * \brief calculate the total number of packets in this frame
     */
    uint32_t Frame_size_in_packet();
}; // class PacketFrame


class PacketSender  : public Object{
public:
    static TypeId GetTypeId (void);
    PacketSender(GameServer * server, uint16_t interval, Time delay_ddl, 
        Ptr<OutputStreamWrapper> debugStream, void (GameServer::*)(Ptr<AckPacket>),
        void (GameServer::*)(Ptr<FrameAckPacket>));
    ~PacketSender();

    void StartApplication(Ptr<Socket> socket);

    void SetController(std::shared_ptr<rmcat::SenderBasedController> controller);

    /**
     * \brief Send packets of a Frame to network, called by GameServer
     * \param packets data and FEC packets of a Frame
     */
    void SendFrame(std::vector<Ptr<VideoPacket>>);

    /**
     * \brief Send retransmission packets
     * \param packets data and FEC packets to be sent immediately
     */
    void SendRtx(std::vector<Ptr<VideoPacket>>);

    /**
     * \brief Caluculate pacing rate and set m_pacing_interval
     */
    void Calculate_pacing_rate();


    void OnSocketRecv_sender(Ptr<Socket>);

    void SetNetworkStatistics(
        Time default_rtt, double_t bw/* in Mbps */,
        double_t loss, double_t group_delay /* in ms */
    );

    void SetNetworkStatisticsBytrace(uint16_t rtt /* in ms */, 
                                    double_t bw/* in Mbps */,
                                    double_t loss_rate);

    Ptr<FECPolicy::NetStat> GetNetworkStatistics();

    double_t GetBandwidthLossRate();

    void UpdateSendingRate(); // apply cc algorithm to calculate the sending bitrate

    double GetSendingRate();

    double GetGoodputRatio();

    void UpdateGoodputRatio();

    void UpdateRTT(Time rtt);

    void UpdateNetstateByTrace();

    void SetTrace(std::string tracefile);

protected:
    void SetTimeLimit(const Time& limit);
    void Pause();
    void Resume();

private:
    /**
     * \brief Send a packet to the network
     */
    void SendPacket();

    /**
     * \brief Burstly send packets to the network
     */
    void SendPacket_burst(std::vector<Ptr<VideoPacket>>);

    /**
     * \brief calculate the number of framees in the queue
     * \return the number of framees in the queue
     */
    uint32_t num_frame_in_queue();

private:
    GameServer * game_server;
    void (GameServer::*ReportACKFunc)(Ptr<AckPacket>);
    void (GameServer::*ReportFrameAckFunc)(Ptr<FrameAckPacket>);
    Ptr<Socket> m_socket; /* UDP socket to send our packets */

    Ptr<FECPolicy::NetStat> m_netStat; /* stats used for FEC para calculation */

    std::vector<Ptr<PacketFrame>> m_queue; /* Queue storing packets by frames */

    //std::vector<Ptr<VideoPacket>> pkts_sent;

    std::unordered_map<uint16_t, Ptr<SentPacketInfo>> pktsHistory; /* Packet history for all packets */
    std::deque<uint16_t> m_send_wnd; // Last ID in sender sliding window
    uint16_t m_send_wnd_size;

    std::deque<uint16_t> m_goodput_wnd;
    uint32_t m_goodput_wnd_size; // in us
    uint64_t goodput_pkts_inwnd;
    uint64_t total_pkts_inwnd;
    float m_goodput_ratio;

    bool m_cc_enable;

    double m_bitrate; // target bitrate calculated by cc algorithm

    uint32_t m_group_id;

    uint64_t m_groupstart_TxTime;

    int m_group_size;

    uint16_t m_prev_id;

    uint64_t m_prev_RxTime;

    uint16_t m_prev_groupend_id;

    uint64_t m_prev_groupend_RxTime;

    int m_prev_group_size;

    uint16_t m_prev_pkts_in_frame;
    uint16_t m_curr_pkts_in_frame;
    uint64_t m_prev_frame_TxTime; // in us
    uint64_t m_curr_frame_TxTime;
    uint64_t m_prev_frame_RxTime;
    uint64_t m_curr_frame_RxTime;

    uint16_t m_interval;

    int64_t inter_arrival;
    uint64_t inter_departure;
    int64_t  inter_delay_var;
    int      inter_group_size;


    bool m_firstFeedback;

    std::shared_ptr<rmcat::SenderBasedController> m_controller;

    /* pacing-related variables */
    bool m_pacing; // whether to turn on pacing

    Time m_pacing_interval; /* the time interval before calling the next SendPacket() */

    Timer m_pacingTimer;

    Time m_send_time_limit;

    EventId m_sendevent;

    EventId m_settrace_event;

    uint16_t m_netGlobalId;
    uint16_t m_dataGlobalId;

    uint16_t m_last_acked_global_id;

    Time m_delay_ddl;
    uint64_t m_finished_frame_cnt;
    uint64_t m_timeout_frame_cnt;

    /* statistics for bandwidth loss rate */
    uint64_t init_data_pkt_count, other_pkt_count;
    uint64_t init_data_pkt_size, other_pkt_size;
    uint64_t exclude_head_init_data_pkt_count, exclude_head_other_pkt_count;
    uint64_t exclude_head_init_data_pkt_size, exclude_head_other_pkt_size;

    /* statistics setup by trace */
    bool trace_set;
    std::string trace_filename;

    Ptr<OutputStreamWrapper> m_debugStream;
};  // class PacketSender


};  // namespace ns3

#endif  /* PACKET_SENDER_H */