#ifndef VIDEO_ENCODER_H
#define VIDEO_ENCODER_H

#include "common-header.h"
#include "fec-policy.h"
#include "ns3/simulator.h"
#include "ns3/nstime.h"
#include <vector>

namespace ns3 {

class GameServer;

/**
 * \brief Video encoder base class. Call packet sender when a frame is encoded.
 *
 */
class VideoEncoder : public Object {
public:
    static TypeId GetTypeId (void);
    /**
     * @brief A encodes video and send frame data to GameServer
     *
     * @param game_server GameServer pointer
     * @param fps frame per second
     */
    VideoEncoder(uint8_t, GameServer *, void (GameServer::*SendFrameFunc)(uint8_t *, uint32_t));
    VideoEncoder();
    ~VideoEncoder();

protected:
    GameServer * game_server;
    void (GameServer::*SendFrameFunc)(uint8_t *, uint32_t);
    uint8_t fps;
    uint32_t bitrate; // in kbps
    EventId encode_event;
    /**
     * \brief Encode a frame and call GameServer
     *
     */
    virtual void EncodeFrame() = 0;
    void ScheduleNext();
public:
    void SetFPS(uint8_t);
    void SetBitrate(uint32_t);
    void StartEncoding();
    void StopEncoding();
    uint32_t GetBitrate();  // in kbps
};  // class VideoEncoder

class DumbVideoEncoder : public VideoEncoder {
public:
    static TypeId GetTypeId (void);
    /**
    static TypeId GetTypeId (void);
     * @brief A Dump encoder whose encoded frames are of the same size
     *
     * @param game_server GameServer pointer
     * @param fps frame per second
     * @param bitrate in kbps
     */
    DumbVideoEncoder(uint8_t fps, uint32_t bitrate, GameServer *, void (GameServer::*SendFrameFunc)(uint8_t *, uint32_t));
    DumbVideoEncoder();
    ~DumbVideoEncoder();
private:
    uint64_t frame_count;
    /**
     * \brief Encode a frame every  1.0 / fps sec.
     *
     */
    virtual void EncodeFrame();
};  // class VideoEncoder

};  // namespace ns3

#endif  /* VIDEO_ENCODER_H */