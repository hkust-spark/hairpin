#ifndef GAME_CLIENT_H
#define GAME_CLIENT_H

#include "common-header.h"
#include "fec-policy.h"
#include "packet-receiver.h"
#include "video-decoder.h"
#include "ns3/application.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/simulator.h"
#include "ns3/assert.h"
#include "ns3/socket.h"
#include <unordered_map>

namespace ns3 {

class GameClient : public Application {

public:
    static TypeId GetTypeId (void);
    GameClient();
    //GameClient(uint8_t fps, Time delay_ddl);
    ~GameClient();
    void Setup(Ipv4Address srcIP, uint16_t srcPort,uint16_t destPort, uint8_t fps, Time delay_ddl, 
        uint32_t wndsize, uint16_t rtt, Ptr<OutputStreamWrapper> appStream, Ptr<OutputStreamWrapper> debugStream);

protected:
    void DoDispose(void);

private:
    void StartApplication(void);
    void StopApplication(void);

private:
    Ptr<PacketReceiver> receiver;
    Ptr<VideoDecoder> decoder;
    Ptr<Socket> m_socket;
    Ptr<FECPolicy> policy;

    uint8_t fps;        /* video fps */
    Time delay_ddl;     /* delay ddl */
    Time rtt;
    Time default_rtt;

    Ipv4Address m_peerIP;
    uint16_t m_peerPort;
    uint16_t m_localPort;

    uint32_t m_receiver_window; /* size of receiver's sliding window (in ms) */

    std::unordered_map<uint32_t, Ptr<PacketGroup>> incomplete_groups;
    std::unordered_map<uint32_t, Ptr<PacketGroup>> complete_groups;
    std::unordered_map<uint32_t, Ptr<PacketGroup>> timeout_groups;
    // measure rtt
    static const uint8_t RTT_WINDOW_SIZE = 10;
    std::deque<uint64_t> rtt_window;    /* in us */
    std::deque<Ptr<GroupPacketInfo>> rtx_history_key;     /* in the order of time */
    std::unordered_map<
        uint32_t, std::unordered_map<uint16_t, 
            std::unordered_map<uint8_t, Time>>> rtx_history;  /* {GroupId: {pkt_id: {tx_count : send_time, ...}, ...}, ...} */

    /* statistics */
    uint64_t rcvd_pkt_cnt;
    uint64_t rcvd_data_pkt_cnt;
    uint64_t rcvd_fec_pkt_cnt;
    std::map<uint8_t, uint64_t> rcvd_pkt_rtx_count;
    std::map<uint8_t, uint64_t> rcvd_group_rtx_count;
    std::map<uint8_t, uint64_t> rcvd_datapkt_rtx_count;
    std::map<uint8_t, uint64_t> rcvd_fecpkt_rtx_count;
    std::map<uint8_t, uint64_t> rcvd_rtxpkt_rtx_count;

    Ptr<OutputStreamWrapper> m_debugStream;

    void InitSocket();
    void OutputStatistics();
    /**
     * @brief Reply ACK to server for recently received packets
     */
    void ReplyACK(std::vector<Ptr<DataPacket>> , uint16_t );

    /**
     * @brief Reply Frame ACK for DMR calculation
     */
    void ReplyFrameACK(uint32_t, Time);

public:
    Ptr<Socket> GetSocket();
    void ReceivePacket(Ptr<VideoPacket> pkt);

};  // class GameClient

};  // namespace ns3

#endif  /* GAME_CLIENT_H */