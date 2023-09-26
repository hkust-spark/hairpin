#include "video-encoder.h"

namespace ns3 {
NS_LOG_COMPONENT_DEFINE("VideoEncoder");

TypeId VideoEncoder::GetTypeId() {
    static TypeId tid = TypeId ("ns3::VideoEncoder")
        .SetParent<Object> ()
        .SetGroupName("bitrate-ctrl")
    ;
    return tid;
};

VideoEncoder::VideoEncoder(uint8_t fps, GameServer * game_server, void (GameServer::*SendFrameFunc)(uint8_t *, uint32_t)) {
    this->game_server = game_server;
    this->SendFrameFunc = SendFrameFunc;
    this->SetFPS(fps);
};

 VideoEncoder::VideoEncoder() {

 };

 VideoEncoder::~VideoEncoder() {

 };

void VideoEncoder::SetFPS(uint8_t fps) {
    this->fps = fps;
};

void VideoEncoder::SetBitrate(uint32_t bitrate_kbps){
    this->bitrate = bitrate_kbps;
};

void VideoEncoder::ScheduleNext() {
    Time next_encoding = MicroSeconds(1e6 / this->fps);
    this->encode_event = Simulator::Schedule(next_encoding, &VideoEncoder::EncodeFrame, this);
};

void VideoEncoder::StartEncoding() {
    this->encode_event = Simulator::ScheduleNow(&VideoEncoder::EncodeFrame, this);
};

void VideoEncoder::StopEncoding() {
    this->encode_event.Cancel();
};

uint32_t VideoEncoder::GetBitrate() {
    return this->bitrate;
};

TypeId DumbVideoEncoder::GetTypeId() {
    static TypeId tid = TypeId ("ns3::DumbVideoEncoder")
        .SetParent<VideoEncoder> ()
        .SetGroupName("bitrate-ctrl")
        .AddConstructor<DumbVideoEncoder> ()
    ;
    return tid;
};

DumbVideoEncoder::DumbVideoEncoder(uint8_t fps, uint32_t bitrate, GameServer * game_server, void (GameServer::*SendFrameFunc)(uint8_t *, uint32_t)) :
    VideoEncoder(fps, game_server, SendFrameFunc) {
    this->bitrate = bitrate;
    this->frame_count = 0;
};

DumbVideoEncoder::DumbVideoEncoder() {};

DumbVideoEncoder::~DumbVideoEncoder() {};

void DumbVideoEncoder::EncodeFrame() {
    uint32_t frame_size = this->bitrate * 1000 / 8 / this->fps;
    // uint8_t * buffer = new uint8_t[frame_size];
    // std::fill_n(buffer, frame_size, 0xff);
    ((this->game_server)->*SendFrameFunc)(nullptr, frame_size);
    this->ScheduleNext();
    this->frame_count ++;
    // if(this->frame_count % (this->fps * 10) == 0)
    //     DEBUG(Simulator::Now().GetSeconds());
};
}