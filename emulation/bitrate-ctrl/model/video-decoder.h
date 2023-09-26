#ifndef VIDEO_DECODER_H
#define VIDEO_DECODER_H

#include "common-header.h"
#include "fec-policy.h"
#include "ns3/object.h"
#include "ns3/network-module.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>

namespace ns3 {

class GameClient;

class VideoFrame : public Object {
    uint16_t data_pkt_num;
    Time encode_time;
    Time first_pkt_rcv_time, last_pkt_rcv_time;
    std::unordered_set<uint16_t> group_ids;
    std::unordered_set<uint16_t> pkts;
public:
    static TypeId GetTypeId (void);
    VideoFrame(Ptr<DataPacket> pkt);
    ~VideoFrame();
    void AddPacket(Ptr<DataPacket> pkt);
    uint16_t GetDataPktNum();
    uint16_t GetDataPktRcvedNum();
    std::unordered_set<uint16_t> GetGroupIds();
    Time GetFrameDelay();
    Time GetEncodeTime();
    Time GetLastRcvTime();
};

class VideoDecoder : public Object {
public:
    static TypeId GetTypeId (void);
    VideoDecoder ();
    VideoDecoder (Time, GameClient *, Ptr<OutputStreamWrapper>, void (GameClient::*)(uint32_t, Time));
    ~VideoDecoder ();

private:
    Time m_delayDdl;
    std::unordered_map<uint32_t, Ptr<VideoFrame>> m_unplayedFrames;
    std::unordered_map<uint32_t, Ptr<VideoFrame>> m_playedFrames;
    uint32_t m_curMinFrameId;
    uint32_t m_curMaxFrameId;

    GameClient * m_gameClient;
    void (GameClient::*m_funcReplyFrameAck)(uint32_t, Time);

    Ptr<OutputStreamWrapper> m_appStream;
public:
    void DecodeDataPacket(std::vector<Ptr<DataPacket>> pkts);
    double_t GetDDLMissRate();

}; // class VideoDecoder

}; // namespace ns3

#endif  /* VIDEO_DECODER_H */