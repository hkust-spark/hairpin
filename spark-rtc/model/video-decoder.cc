#include "video-decoder.h"

namespace ns3 {
NS_LOG_COMPONENT_DEFINE("VideoDecoder");

TypeId VideoFrame::GetTypeId() {
    static TypeId tid = TypeId ("ns3::VideoFrame")
        .SetParent<Object> ()
        .SetGroupName("spark-rtc")
    ;
    return tid;
};

VideoFrame::VideoFrame(Ptr<DataPacket> pkt) {
    this->group_ids.insert(pkt->GetGroupId());
    this->data_pkt_num = pkt->GetFramePktNum();
    this->first_pkt_rcv_time = pkt->GetRcvTime();
    this->last_pkt_rcv_time = pkt->GetRcvTime();
    this->encode_time = pkt->GetEncodeTime();
    this->AddPacket(pkt);
};

VideoFrame::~VideoFrame() {};

void VideoFrame::AddPacket(Ptr<DataPacket> pkt) {
    if(pkt->GetRcvTime() > this->last_pkt_rcv_time)
        this->last_pkt_rcv_time = pkt->GetRcvTime();
    if(pkt->GetEncodeTime() < this->encode_time)
        this->encode_time = pkt->GetEncodeTime();
    this->pkts.insert(pkt->GetPktIdFrame());
};

uint16_t VideoFrame::GetDataPktNum() { return this->data_pkt_num; };

uint16_t VideoFrame::GetDataPktRcvedNum() { return this->pkts.size(); };

std::unordered_set<uint16_t> VideoFrame::GetGroupIds() { return this->group_ids; };

Time VideoFrame::GetFrameDelay() {
    return last_pkt_rcv_time - encode_time;
};

Time VideoFrame::GetEncodeTime() {
    return this->encode_time;
};

Time VideoFrame::GetLastRcvTime() {
    return this->last_pkt_rcv_time;
}


TypeId VideoDecoder::GetTypeId() {
    static TypeId tid = TypeId ("ns3::VideoDecoder")
        .SetParent<Object> ()
        .SetGroupName("spark-rtc")
        .AddConstructor<VideoDecoder> ()
    ;
    return tid;
};

VideoDecoder::VideoDecoder (Time delayDdl, GameClient * gameClient, Ptr<OutputStreamWrapper> appStream,
    void (GameClient::*ReplyFrameAck)(uint32_t, Time)) {
    m_delayDdl = delayDdl;
    m_curMinFrameId = 0;
    m_curMaxFrameId = 0;
    m_gameClient = gameClient;
    m_appStream = appStream;
    m_funcReplyFrameAck = ReplyFrameAck;
};

VideoDecoder::VideoDecoder () {

};

VideoDecoder::~VideoDecoder () {

};

void VideoDecoder::DecodeDataPacket (std::vector<Ptr<DataPacket>> pkts) {
    Time now = Simulator::Now();
    for(auto pkt : pkts) {
        uint32_t frameId = pkt->GetFrameId();
        if (m_playedFrames.find (frameId) != m_playedFrames.end ()) {
            continue;
        }
        
        m_curMaxFrameId = std::max (m_curMaxFrameId, frameId);
        m_curMinFrameId = std::min (m_curMinFrameId, frameId);

        if (m_unplayedFrames.find (frameId) == m_unplayedFrames.end ())
            m_unplayedFrames[frameId] = Create<VideoFrame> (pkt);
        else
            m_unplayedFrames[frameId]->AddPacket (pkt);

        if (m_unplayedFrames[frameId]->GetDataPktNum ()
            == m_unplayedFrames[frameId]->GetDataPktRcvedNum ()) {
            m_playedFrames[frameId] = m_unplayedFrames [frameId];
            m_unplayedFrames.erase (frameId);
            (m_gameClient->*m_funcReplyFrameAck) (frameId, m_playedFrames[frameId]->GetEncodeTime ());
            *m_appStream->GetStream () << "Frame " << frameId << 
                " encoded " << m_playedFrames[frameId]->GetEncodeTime ().GetMilliSeconds () << 
                " played at " << now.GetMilliSeconds () << 
                " missddl? " << (m_playedFrames[frameId]->GetFrameDelay () > m_delayDdl) << std::endl;
            NS_ASSERT (m_playedFrames[frameId]->GetFrameDelay () == now - m_playedFrames[frameId]->GetEncodeTime ());
        }
    }
};

double_t VideoDecoder::GetDDLMissRate() {
    /* ddl miss rate = missed_frames / all_frames */
    uint64_t frame_rcvd_cnt = m_playedFrames.size(),
        frame_total_cnt = m_curMaxFrameId - m_curMinFrameId + 1;
    // std::cout << "Group id of unplayed frames: ";
    // for(auto it = m_unplayedFrames.begin();it != m_unplayedFrames.end();it ++) {
    //     std::cout << it->first << ": ";
    //     for(auto id : it->second->GetGroupIds()) std::cout << id << ", ";
    // }
    // std::cout);
    // for(auto it = m_playedFrames.begin();it != m_playedFrames.end();it ++) {
    //     if(it->second->GetFrameDelay() > this->delay_ddl * 1.05) {
    //         frame_loss_count ++;
    //         // DEBUG("[VideoDecoder] Frame miss. Encode time: " << it->second->GetEncodeTime().GetMilliSeconds() << ", last rcv time: " << it->second->GetLastRcvTime().GetMilliSeconds() << ", Group id: ";
    //         auto group_ids = it->second->GetGroupIds();
    //         // for(auto i = group_ids.begin();i != group_ids.end();i++)
    //             // std::cout << *i << ", ";
    //         // std::cout << std::endl;
    //         NS_LOG_FUNCTION("lost frame id "<<it->first);
    //     }
    // }
    if(frame_total_cnt == 0)  return 0;
    double_t ddl_miss_rate = ((double_t) frame_total_cnt - frame_rcvd_cnt) / frame_total_cnt;
    NS_LOG_ERROR("[Decoder] Total frames: " << frame_total_cnt << ", played frames: " << m_playedFrames.size() << ", unplayed frames: " << m_unplayedFrames.size());
    NS_LOG_ERROR("[Decoder] DDL Miss Rate: " << ddl_miss_rate * 100 << "%");

    return ddl_miss_rate;
};
};