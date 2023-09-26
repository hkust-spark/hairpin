#ifndef GAME_SERVER_H
#define GAME_SERVER_H

#include "common-header.h"
#include "packet-sender.h"
#include "video-encoder.h"
#include "fec-policy.h"
#include "other-policy.h"
#include "ns3/application.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/simulator.h"
#include "ns3/assert.h"
#include "ns3/socket.h"
#include <vector>
#include <deque>
#include <unordered_map>

namespace ns3 {

class PacketSender;

enum CC_ALG { NOT_USE_CC, GCC, NADA, SCREAM};

class LossEstimator : public Object {
public:
    LossEstimator (Time window);
    ~LossEstimator ();
    static TypeId GetTypeId (void);
private:
    std::deque<std::pair<uint16_t, Time>> m_sendList;
    std::deque<std::pair<uint16_t, Time>> m_rtxList;
    Time m_window;
public:
    void SendUpdate (uint16_t num, Time now);
    void RtxUpdate (uint16_t num, Time now);
    double_t GetLoss (Time now);
};

class GameServer : public Application {

/* Override Application */
public:
    static TypeId GetTypeId (void);
    /**
     * \brief Construct a new GameServer object
     *
     * \param fps Default video output config: frames per second
     * \param delay_ddl
     * \param bitrate in Kbps
     * \param pacing_flag
     */
    GameServer();
    //GameServer(uint8_t, Time, uint32_t, bool);
    ~GameServer();
    void Setup(
        Ipv4Address srcIP, uint16_t srcPort, Ipv4Address destIP, uint16_t destPort,
        uint8_t fps, Time delay_ddl, uint32_t bitrate, uint16_t interval,
        Ptr<FECPolicy> fecPolicy, std::string rtxPolicy, Time measure_window,
        uint16_t default_rtt /* in ms */, double_t default_bw/* in Mbps */,
        double_t default_loss, double_t default_group_delay /* in ms */,
        bool trace_set, std::string trace_file, Ptr<OutputStreamWrapper> fecStream,
        Ptr<OutputStreamWrapper> debugcStream
    );
    void SetController(CC_ALG);
    void StopEncoding();
protected:
    void DoDispose();
private:
    void StartApplication();
    void StopApplication();

/* Packet sending logic */
private:
    Ptr<PacketSender> m_sender;       /* Pacing, socket operations */
    Ptr<VideoEncoder> m_encoder;      /* Provides encoded video data */
    Ptr<FECPolicy> m_fecPolicy;          /* FEC Policy */
    std::string m_rtxPolicy;          /* FEC Policy */
    Ptr<Socket> m_socket;           /* UDP Socket */

    class UnFECedPackets : public Object {
    public:
        FECPolicy::FECParam param;
        uint16_t next_pkt_id_in_batch;
        uint16_t next_pkt_id_in_group;
        std::vector<Ptr<DataPacket>> pkts;

        static TypeId GetTypeId (void);
        UnFECedPackets();
        ~UnFECedPackets();
        void SetUnFECedPackets(
            FECPolicy::FECParam param,
            uint16_t next_pkt_id_in_batch, uint16_t next_pkt_id_in_group,
            std::vector<Ptr<DataPacket>> pkts
        );
    };


    std::deque<Ptr<GroupPacketInfo>> m_dataPktHistoryKey;     /* in time order */

    /* (GroupId, pkt_id_group) -> Packet */
    std::unordered_map<
        uint32_t, std::map<uint16_t, Ptr<DataPacket>>> m_dataPktHistory;    /* Packets sent in the past ddl, temporary */
    std::unordered_map<
        uint32_t, std::deque<uint32_t>> m_frameIdToGroupId; /* frame_id -> group_id */
    /* Record how many data packets there are in a frame */
    /* frame id -> data packet count */
    std::unordered_map<uint32_t, uint8_t> m_frameDataPktCnt;
    uint16_t m_curRxHighestDataGlobalId;
    uint16_t m_curRxHighestGlobalId;
    uint16_t m_curContRxHighestGlobalId;
    bool m_isRecovery;
    std::unordered_map<uint32_t, Time> m_delayedRtxGroup;
    Time m_lastRtt;

    uint32_t m_nextFrameId;  /* accumulated */
    uint32_t m_nextGroupId;  /* accumulated */
    uint32_t m_nextBatchId;  /* for simplicity, batch_id is also globally incremental */
    uint8_t fps;        /* video fps */
    uint32_t bitrate;   /* in Kbps */
    uint16_t m_frameInterval;  /* ms, pass the info to packet sender */
    Time m_delayDdl;     /* delay ddl */
    bool pacing_flag;   /* flag for pacing */

    Time check_rtx_interval;
    EventId check_rtx_event; /* Timer for retransmisstion */
    Ptr<LossEstimator> m_lossEstimator;

    Ipv4Address m_srcIP;
    uint16_t m_srcPort;
    Ipv4Address m_destIP;
    uint16_t m_destPort;

    /* Congestion Control-related variables */
    bool m_cc_enable;
    CC_ALG m_cc_algorithm;
    Timer m_cc_timer;
    Time m_cc_interval;

    int m_ccaQuotaPkt;

    FECPolicy::FECParam m_curFecParam; /* record the latest FEC parameters for encoding bitrate convertion */

    float m_goodput_ratio; /* (data pkt number / all pkts sent) in a time window */

    uint32_t m_maxPayloadSize;

    // statistics
    uint64_t send_group_cnt;
    uint64_t send_frame_cnt;

    Ptr<OutputStreamWrapper> m_fecStream;
    Ptr<OutputStreamWrapper> m_debugStream;

    /**
     * \brief Initialize a UDP socket
     *
     */
    void InitSocket();

    uint32_t GetNextGroupId();
    uint32_t GetNextBatchId();
    uint32_t GetNextFrameId();

    /**
     * \brief Create a packet batch from pkt_batch
     *
     * \param pkt_batch Data packets for creating the batch, store the whole batch afterwards
     * \param fec_param FEC parameters
     * \param new_group flag to create a new group
     * \param group_id if new_group==false, use group_id instead of creating a new id
     * \param is_rtx
     */
    void CreatePacketBatch(
        std::vector<Ptr<VideoPacket>>& pkt_batch,
        FECPolicy::FECParam fec_param,
        bool new_group, uint32_t group_id, bool is_rtx
    );

    void CreateFirstPacketBatch(
    std::vector<Ptr<VideoPacket>>& pkt_batch, FECPolicy::FECParam fec_param);
    void CreateRTXPacketBatch(
    std::vector<Ptr<VideoPacket>>& pkt_batch, FECPolicy::FECParam fec_param);

    /**
     * \brief For incomplete group/batch, complete it with new data packets
     *
     * \param new_data_pkts new data packets for creating the batch, store the whole batch afterwards
     * \param fec_pkts Store FEC packets
     */
    void CompletePacketBatch(
        std::vector<Ptr<VideoPacket>>& new_data_pkts, std::vector<Ptr<VideoPacket>>& fec_pkts
    );

    /**
     * \brief Determine whether the packet will definitely exceed delay_ddl
     *
     * \param pkt
     *
     * \return true it will exceed delay_ddl
     * \return false it probably will not
     */
    bool MissesDdl(Ptr<DataPacket> packet);

    Time GetDispersion (Ptr<DataPacket> pkt);

    bool IsRtxTimeout (Ptr<DataPacket> packet, Time rto);


    /**
     * \brief Store data packets for retransmission
     *
     * \param pkts
     */
    void StorePackets(std::vector<Ptr<VideoPacket>> pkts);

    void SendPackets(std::deque<Ptr<DataPacket>> pkts, Time ddl_left, uint32_t frame_id, bool is_rtx);

    void RetransmitGroup(uint32_t);

    /**
     * \brief Check for packets to be retransmitted regularly
     *
     */
    void CheckRetransmission();

    FECPolicy::FECParam GetFECParam(
        uint16_t max_group_size, 
        uint32_t bitrate,
        Time ddl_left, bool fix_group_size,
        bool is_rtx, uint8_t frame_size
    );

public:
    /**
     * \brief A frame of encoded data is provided by encoder. Called every frame.
     *
     * \param buffer Pointer to the start of encoded data
     * \param size Length of the data in bytes
     */
    void SendFrame(uint8_t * buffer, uint32_t size);

    /**
     * \brief Process ACK packet
     *
     * \param pkt contains info of the packets that have been received
     */
    void RcvACKPacket(Ptr<AckPacket> pkt);

    void RcvFrameAckPacket (Ptr<FrameAckPacket> frameAckPkt);

    /**
     * \brief For packet_sender to get the UDP socket
     *
     * \return Ptr<Socket> a UDP socket
     */
    Ptr<Socket> GetSocket();

    /**
     * @brief Update sending bitrate in CC controller and report to video encoder.
     *
     */
    void UpdateBitrate();

    /**
     * @brief Print group info
     */
    void OutputStatistics();

};  // class GameServer

};  // namespace ns3

#endif  /* GAME_SERVER_H */