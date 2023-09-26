/******************************************************************************
 * Copyright 2016-2017 Cisco Systems, Inc.                                    *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License");            *
 * you may not use this file except in compliance with the License.           *
 *                                                                            *
 * You may obtain a copy of the License at                                    *
 *                                                                            *
 *     http://www.apache.org/licenses/LICENSE-2.0                             *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 ******************************************************************************/

/**
 * @file
 * Partial implementation of abstract class representing the interface to a
 * sender-based controller.
 *
 * @version 0.1.1
 * @author Jiantao Fu
 * @author Sergio Mena
 * @author Xiaoqing Zhu
 */
#include "sender-based-controller.h"
#include <numeric>
#include <iostream>
#include <sstream>
#include <vector>
#include <cassert>


namespace rmcat {

const int MIN_PACKET_LOGLEN = 1;             /**< minimum # of packets in log for stats to be meaningful */
const uint64_t MAX_INTER_PACKET_TIME_US = 500 * 1000;  /**< maximum interval between packets, in microseconds */
const uint64_t DEFAULT_HISTORY_LENGTH_US = 500 * 1000; /**< default time window for logging history of packets, in microseconds */
const float RMCAT_CC_DEFAULT_RINIT = 1000000.; /**< Initial BW in bps: 150Kbps */
const float RMCAT_CC_DEFAULT_RMIN = 1000000.;  /**< in bps: 1Mbps */
const float RMCAT_CC_DEFAULT_RMAX = 30000000.; /**< in bps: 30Mbps */

InterLossState::InterLossState()
: intervals{}
, expectedSeq{0}
, initialized{false}
{
    intervals.push_front(0);
}

void SenderBasedController::setDefaultId() {
    // By default, the id is the object's address
    std::stringstream ss;
    ss << this;
    m_id = ss.str();
}

SenderBasedController::SenderBasedController()
: m_firstSend{true},
  m_lastSequence{0},
  m_baseDelayUs{0},
  m_inTransitPackets{},
  m_PacketTransitHistory{},
  m_packetHistory{},
  m_recvHistory{},
  m_pktSizeSum{0},
  m_lastRecvSequence{0},
  m_id{},
  m_initBw{RMCAT_CC_DEFAULT_RINIT},
  m_minBw{RMCAT_CC_DEFAULT_RMIN},
  m_maxBw{RMCAT_CC_DEFAULT_RMAX},
  m_logCallback{NULL},
  m_ilState{},
  m_lost{0},
  last_fraction_loss_{0},
  loss_counter{0},
  m_historyLengthUs{DEFAULT_HISTORY_LENGTH_US} {
      setDefaultId();
}

SenderBasedController::~SenderBasedController() {}

void SenderBasedController::setId(const std::string& id) {
    m_id = id;
}

void SenderBasedController::setInitBw(float initBw) {
    m_initBw = initBw;
}

void SenderBasedController::setMinBw(float minBw) {
    m_minBw = minBw;
}

void SenderBasedController::setMaxBw(float maxBw) {
    m_maxBw = maxBw;
}

void SenderBasedController::setLogCallback(logCallback f) {
    m_logCallback = f;
}

void SenderBasedController::reset() {
    m_firstSend = true;
    m_lastSequence = 0;
    m_baseDelayUs = 0;
    m_inTransitPackets.clear();
    m_recvHistory.clear();
    m_pktSizeSum = 0;
    m_initBw = RMCAT_CC_DEFAULT_RINIT;
    m_minBw = RMCAT_CC_DEFAULT_RMIN;
    m_maxBw = RMCAT_CC_DEFAULT_RMAX;
    m_logCallback = NULL;
    m_ilState = InterLossState{};
    m_historyLengthUs = DEFAULT_HISTORY_LENGTH_US;
    setDefaultId();
}

//TODO (deferred): This logic is to be encapsulated within class InterLossState
void SenderBasedController::updateInterLossData(const PacketRecord& packet) {
    if (m_packetHistory.empty()) {
        m_ilState = InterLossState{}; // Reset
        m_ilState.expectedSeq = packet.sequence;
    }

    // update state for TFRC-style inter-loss interval calculation
    if (packet.sequence == m_ilState.expectedSeq) {
        ++m_ilState.intervals[0];
        ++m_ilState.expectedSeq;
        return;
    }
    assert(lessThan(m_ilState.expectedSeq,packet.sequence));
    m_ilState.intervals.push_front(1); // Start new interval; shift the existing ones
    if (m_ilState.intervals.size() > 9) {
        m_ilState.intervals.pop_back();
    }

    m_ilState.expectedSeq = packet.sequence + 1;
    m_ilState.initialized = true;
}

bool SenderBasedController::processSendPacket(uint64_t txTimestampUs,
                                              uint16_t sequence,
                                              uint32_t size) {
    if (m_firstSend) {
        m_lastSequence = sequence - 1;
        m_firstSend = false;
    }

    ++m_lastSequence;
    
    if (sequence != m_lastSequence) {
        std::cerr << "SenderBasedController::ProcessSendPacket,"
                  << " illegal sequence: " << sequence
                  << ", should be " << m_lastSequence << std::endl;
        return false;
    }

    // DEBUG(m_lastSequence << " " << txTimestampUs << " " << size);

    // record sent packets in local record
    m_inTransitPackets.push_back(PacketRecord{m_lastSequence,
                                              txTimestampUs,
                                              size,
                                              0,
                                              0});
    // Record all sent packets.
    m_PacketTransitHistory.push_back(PacketRecord{m_lastSequence,
                                                  txTimestampUs,
                                                  size,
                                                  0,
                                                  0});
    // Memory safety: timestamps of in-transit packets must be
    //  within (10 * MAX_INTER_PACKET_TIME)
    while (true) {
        const uint64_t firstTimestampUs = m_inTransitPackets.front().txTimestampUs;
        if (lessThan(firstTimestampUs + 10 * MAX_INTER_PACKET_TIME_US,
                     txTimestampUs)) {
            m_inTransitPackets.pop_front();
        } else {
            break;
        }
    }
    return true;
}

bool SenderBasedController::processFeedback(uint64_t nowUs,
                                            uint16_t sequence,
                                            uint64_t rxTimestampUs,
                                            int64_t l_inter_arrival,
                                            uint64_t l_inter_departure,
                                            int64_t l_inter_delay_var,
				                            int l_inter_group_size, 
                                            int64_t l_arrival_time,
					                        uint8_t ecn) {
    if (lessThan(m_lastSequence, sequence)) {
        std::cerr << "SenderBasedController::ProcessFeedback,"
                  << " strange sequence: " << sequence
                  << " from the future" << std::endl;
        return false;
    }

    if (m_inTransitPackets.empty()) {
        std::cerr << "SenderBasedController::ProcessFeedback,"
                  << " sequence: " << sequence
                  << " duplicate or out of order" << std::endl;
        // Returning true because it is considered valid to process
        // duplicate/out of order sequences
        return true;
    }

    assert(m_inTransitPackets.back().sequence == m_lastSequence);

    m_lost = 0;
    while (lessThan(m_inTransitPackets.front().sequence, sequence)) {
        // Packet lost or out of order. Remove stale entry
        m_lost++;
		loss_counter++;
        m_inTransitPackets.pop_front();
        // Note: we can't tell whether the media (forward path) packet
        //     or the feedback (backward path) packet was lost.
        // Assuming media packet was lost for the time being
    }
    
    /*while (lessThan(m_PacketTransitHistory.front().sequence, sequence)) {
        m_PacketTransitHistory.pop_front();
    }*/

    if (lessThan(sequence, m_inTransitPackets.front().sequence)) {
        std::cerr << "SenderBasedController::ProcessFeedback,"
                  << " sequence: " << sequence
                  << " out of order" << std::endl;
        return true;
    }

    PacketRecord packet = m_inTransitPackets.front();
    m_inTransitPackets.pop_front();
    assert(sequence == packet.sequence);

    if (!m_packetHistory.empty()) {
        const PacketRecord& lastPacket = m_packetHistory.back();
        if (lessThan(packet.txTimestampUs, lastPacket.txTimestampUs)) {
            std::cerr << "SenderBasedController::ProcessFeedback,"
                      << " sequence: " << sequence
                      << " has decreasing timestamp " << packet.txTimestampUs
                      << " w.r.t. sequence " << lastPacket.sequence
                      << " with timestamp " << lastPacket.txTimestampUs
                      << std::endl;
            return false;
        }
        if (lessThan(lastPacket.txTimestampUs + MAX_INTER_PACKET_TIME_US,
                     packet.txTimestampUs)) {
            // It's been too long without receiving any feedback packet
            // Packet history is obsolete
            m_packetHistory.clear();
            m_pktSizeSum = 0;
        }
    }

    // Sanity check
    assert(packet.owdUs == 0);
    assert(packet.rttUs == 0);

    // This subtraction can wrap if clocks aren't synchronized, but it's OK
    packet.owdUs = rxTimestampUs - packet.txTimestampUs;
    packet.rttUs = nowUs - packet.txTimestampUs;

    if (m_packetHistory.empty() || lessThan(packet.owdUs, m_baseDelayUs)) {
        m_baseDelayUs = packet.owdUs;
    }

    updateInterLossData(packet);

    m_packetHistory.push_back(packet);
    m_recvHistory.push_back(packet);
    m_pktSizeSum += packet.size;

    // Garbage collect history to keep its length within limits
    while (true) {
        const uint64_t lastTimestampUs = m_packetHistory.back().txTimestampUs;
        const uint64_t firstTimestampUs = m_packetHistory.front().txTimestampUs;
        assert (!lessThan(lastTimestampUs, firstTimestampUs));
        if (lessThan(lastTimestampUs, firstTimestampUs + m_historyLengthUs)) {
            break;
        }
        const uint32_t firstSize = m_packetHistory.front().size;
        m_packetHistory.pop_front();
        assert(m_pktSizeSum >= firstSize);
        m_pktSizeSum -= firstSize;
    }
    return true;
}

void SenderBasedController::UpdateLossRate(uint8_t loss_q8) {
  last_fraction_loss_ = loss_q8;
}

void SenderBasedController::PrunTransitHistory(uint32_t tar_seq) {
    while(m_PacketTransitHistory.front().sequence < tar_seq) {
        // DEBUG("PrunTransitHistory:: " << m_PacketTransitHistory.front().sequence );DEBUG("PrunTransitHistory:: " << m_PacketTransitHistory.front().sequence );
        // std::cout << "PrunTransitHistory:: " << tar_seq << "\n";
        m_PacketTransitHistory.pop_front();
    }
}

uint64_t SenderBasedController::GetPacketTxTimestamp(uint16_t sequence){
    int i = 0;
    int qsize = m_PacketTransitHistory.size();

    // DEBUG("GETPKT::at(0):: " << m_PacketTransitHistory.at(0).sequence);
    for(i = qsize - 1; i >= 0; i--){
        if(m_PacketTransitHistory.at(i).sequence == sequence){
            // DEBUG("GetPacketTxTimestamp:: " << m_PacketTransitHistory.at(i).txTimestampUs); 
            return m_PacketTransitHistory.at(i).txTimestampUs;
        }
    }
    return -1;
}

uint64_t SenderBasedController::UpdateDepartureTime(uint32_t prev_s, uint32_t now_s){
    int qsize = m_PacketTransitHistory.size();
    uint64_t prev_t = 0, now_t = 0;

    int i = 0;
    // DEBUG("UpdateDepartureTime:: " << prev_s << ", " << now_s);DEBUG("UpdateDepartureTime:: " << prev_s << ", " << now_s);
    // std::cout << "UPDATEPKT::at(0):: " << m_PacketTransitHistory.at(0).sequence << "\n";
    for(i = 0; i < qsize; i++){
        
        if(m_PacketTransitHistory.at(i).sequence == prev_s){
	    prev_t = m_PacketTransitHistory.at(i).txTimestampUs;
     //       DEBUG("prev_t:: " << prev_t);
	}
	
	if(m_PacketTransitHistory.at(i).sequence == now_s){
	    now_t = m_PacketTransitHistory.at(i).txTimestampUs;
   //         DEBUG("now_t:: " << now_t);
	}
    }

    return (now_t - prev_t);
}

uint64_t SenderBasedController::GetPacketSize(uint16_t sequence) {
    int i = 0;
    int qsize = m_PacketTransitHistory.size();

    // DEBUG("GETPKT::at(0):: " << m_PacketTransitHistory.at(0).sequence);
    for(i = 0; i < qsize; i++){
        if(m_PacketTransitHistory.at(i).sequence == sequence){
            // DEBUG("GetPacketTxTimestamp:: " << m_PacketTransitHistory.at(i).txTimestampUs); 
            return m_PacketTransitHistory.at(i).size;
        }
    }
    return -1;
}

void SenderBasedController::setHistoryLength(uint64_t lenUs) {
    m_historyLengthUs = lenUs;
}

uint64_t SenderBasedController::getHistoryLength() const {
    return m_historyLengthUs;
}

// These functions calculate different metrics based on the feedback received.
// Although they could be considered part of the NADA algorithm, we have
// defined them in the superclass because they could also be useful to other
// algorithms
bool SenderBasedController::getCurrentQdelay(uint64_t& qdelayUs) const {
    // 15-tab minimum filtering
    const size_t ntab = 15;
    if (m_packetHistory.empty()) {
        std::cerr << "SenderBasedController::getCurrentQdelay,"
                  << " cannot calculate qdelay, packet history is empty"
                  << std::endl;
        return false;
    }

    uint64_t qDelayMinUs = 0;
    size_t iter = 0;
    for (auto rit = m_packetHistory.rbegin();
            rit != m_packetHistory.rend();
            ++rit) {
        const uint64_t qDelayCurrentUs = rit->owdUs - m_baseDelayUs;
        if (iter > 0) {
            qDelayMinUs = std::min(qDelayMinUs, qDelayCurrentUs);
        } else {
            qDelayMinUs = qDelayCurrentUs;
        }
        if (++iter >= ntab) {
            break;
        }
    }
    qdelayUs = qDelayMinUs;
    return true;
}

bool SenderBasedController::getCurrentRTT(uint64_t& rttUs) const {
    // 15-tab minimum filtering
    const size_t ntab = 15;
    if (m_packetHistory.empty()) {
        std::cerr << "SenderBasedController::getCurrentRTT,"
                  << " cannot calculate rtt, packet history is empty"
                  << std::endl;
        return false;
    }

    uint64_t rttMinUs = 0;
    size_t iter = 0;
    for (auto rit = m_packetHistory.rbegin();
            rit != m_packetHistory.rend();
            ++rit) {
        const uint64_t rttCurrentUs = rit->rttUs;
        if (iter > 0) {
            rttMinUs = std::min(rttMinUs, rttCurrentUs);
        } else {
            rttMinUs = rttCurrentUs;
        }
        if (++iter >= ntab) {
            break;
        }
    }

    rttUs = rttMinUs;
    return true;
}

bool SenderBasedController::getPktLossInfo(uint32_t& nLoss, float& plr, uint32_t& nPkt) const {
    if (m_packetHistory.size() < MIN_PACKET_LOGLEN) {
        std::cerr << "SenderBasedController::getPktLossInfo,"
                  << " packet history too short: "
                  << m_packetHistory.size()
                  << " < " << MIN_PACKET_LOGLEN << std::endl;
        return false;
    }

    // This will wrap properly
    const uint16_t seqSpan = 1u + m_packetHistory.back().sequence
                             - m_packetHistory.front().sequence;
    assert(seqSpan >= m_packetHistory.size());
    nLoss = seqSpan - m_packetHistory.size();
    // nLoss = m_lost;
    plr = float(nLoss) / float(seqSpan);
	nPkt = seqSpan;
    return true;
}

bool SenderBasedController::getCurrentRecvRate(float& rrateBps) {
    if (m_packetHistory.size() < MIN_PACKET_LOGLEN) {
        std::cerr << "SenderBasedController::getCurrentRecvRate,"
                  << " packet history too short: "
                  << m_packetHistory.size()
                  << " < " << MIN_PACKET_LOGLEN << std::endl;
        return false;
    }

    const PacketRecord& front = m_packetHistory.front();
    const PacketRecord& back = m_packetHistory.back();
    const uint64_t firstRxUs = front.txTimestampUs + front.owdUs;
    const uint64_t lastRxUs = back.txTimestampUs + back.owdUs;
    // assert(lessThan(firstRxUs, lastRxUs + 1));
    uint64_t timeSpanUs = lastRxUs - firstRxUs;

    if (!lessThan(firstRxUs, lastRxUs + 1) || timeSpanUs == 0) {
        std::cerr << "SenderBasedController::getCurrentRecvRate,"
                  << " cannot calculate receive rate,"
                  << " all packets were received simultaneously" << std::endl;
        return false;
    }

    uint64_t curr = lastRxUs;
    uint32_t pktcnt = 0;
    size_t i = m_recvHistory.size() - 1;

    while ((lastRxUs - curr) < 1000000){
        pktcnt += m_recvHistory.at(i).size;
        curr = m_recvHistory.at(i).txTimestampUs + m_recvHistory.at(i).owdUs;
       
       if(i == 0)
          return false;
        i--;
     }

     for(uint32_t j = 0;j<=i;j++)
        m_recvHistory.pop_front();
    
      // Technically, the first packet is out of the calculated time span
      assert(front.size <= m_pktSizeSum);
      const uint32_t bytes = pktcnt;
      rrateBps = float(bytes * 8) * 1000.f * 1000.f / (lastRxUs - curr);
      // DEBUG("getCUrrentRecvRate:: rrateBps( " << rrateBps);
      return true;
}


bool SenderBasedController::getLossIntervalInfo(float& avgInterval, uint16_t& currentInterval) const {
    if (!m_ilState.initialized) {
        return false; // No losses yet --> no intervals
    }

    const size_t k = m_ilState.intervals.size();
    assert(k >= 2 && k <= 9);

    static std::vector<float> weights{};
    // Values are 1, 1, 1, 1, .8, .6, .4, .2
    if (weights.empty()) {
        const size_t n = 8;
        for (size_t i = 0; i < n; ++i) {
            weights.push_back((i < n / 2) ? 1.f : 2.f * float(n - i) / float(n + 2));
        }
    }

    const float iSum0 = std::inner_product(m_ilState.intervals.begin(),
                                           m_ilState.intervals.end() - 1,
                                           weights.begin(), 0.f);
    const float iSum1 = std::inner_product(m_ilState.intervals.begin() + 1,
                                           m_ilState.intervals.end(),
                                           weights.begin(), 0.f);
    const float wSum = std::accumulate(weights.begin(), weights.begin() + (k - 1), 0.f);
    const float iAvgMax = std::max(iSum0, iSum1) / wSum;
    avgInterval = iAvgMax;
    currentInterval = m_ilState.intervals.front();
    return true;
}

void SenderBasedController::logMessage(const std::string& log) const {
    if (m_logCallback != NULL){
        m_logCallback(log);
    } else {
        std::cout << log << std::endl;
    }
}

}
