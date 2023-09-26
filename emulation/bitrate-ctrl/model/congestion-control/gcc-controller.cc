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
 * Dummy controller (CBR) implementation for rmcat ns3 module.
 *
 * @version 0.1.1
 * @author Jiantao Fu
 * @author Sergio Mena
 * @author Xiaoqing Zhu
 */

#include "ns3/common-header.h"
#include "sender-based-controller.h"
#include "gcc-controller.h"
#include <sstream>
#include <cassert>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <string>
#include <limits>
#include <cstdio>
#include <iostream>
#include <fstream>
#include "ns3/simulator.h" 

#define LOSS_TIMER 1000
namespace rmcat {

enum { kMinFramePeriodHistoryLength = 60 };
enum { kDeltaCounterMax = 1000 };

const double kMaxAdaptOffsetMs = 15.0;
const double kOverUsingTimeThreshold = 10;
const int kMinNumDeltas = 60;

static const int64_t kDefaultRttMs = 200;
static const int64_t kMaxFeedbackIntervalMs = 1000;
static const float kDefaultBackoffFactor = 0.85f;

const int64_t kBweIncreaseIntervalMs = 1000;
const int64_t kBweDecreaseIntervalMs = 300;
const int64_t kStartPhaseMs = 2000;
const int64_t kBweConverganceTimeMs = 20000;
const int kLimitNumPackets = 20;
const int kDefaultMaxBitrateBps = 1000000000;
const int64_t kLowBitrateLogPeriodMs = 10000;
const int64_t kRtcEventLogPeriodMs = 5000;

// Expecting that RTCP feedback is sent uniformly within [0.5, 1.5]s intervals.
const int64_t kFeedbackIntervalMs = 5000;
const int64_t kFeedbackTimeoutIntervals = 3;
const int64_t kTimeoutIntervalMs = 1000;

const float kDefaultLowLossThreshold = 0.02f;
const float kDefaultHighLossThreshold = 0.1f;
const int kDefaultBitrateThresholdKbps = 0;

const size_t kDefaultTrendlineWindowSize = 10;
const double kDefaultTrendlineSmoothingCoeff = 0.9;
const double kDefaultTrendlineThresholdGain = 4.5;

GccController::GccController() :
    SenderBasedController{},
    m_pkt_cnt{0},
    m_lastTimeCalcUs{0},
    m_lastTimeCalcValid{false},
    m_QdelayUs{0},
	  m_Pkt{0},
    m_ploss{0},
    m_plr{0.f},
    m_RecvR{0.}, 
	  m_timer{0},
	  prev_seq_loss{0},
	  loss_moving_avg{0.0},
	  m_plrmoving_avg{0.f},	

    num_of_deltas_(0),
    slope_(8.0/512.0),	//need initial value
    offset_(0),	//need initial value
    prev_offset_(0),	//need initial value
    E_(),
    process_noise_(),
    avg_noise_(0.0),	//need initial value
    var_noise_(50),	//need initial value
    ts_delta_hist_(),	
 
    window_size_(kDefaultTrendlineWindowSize),
    smoothing_coef_(kDefaultTrendlineSmoothingCoeff),
    threshold_gain_(kDefaultTrendlineThresholdGain),
    accumulated_delay_(0),
    smoothed_delay_(0),
    delay_hist_(),
    
    k_up_(0.0087),
    k_down_(0.039),
    overusing_time_threshold_(kOverUsingTimeThreshold),
    threshold_(12.5),
    last_update_ms_(-1),
    ut_last_update_ms_(-1),
    D_prev_offset_(0.0),
    time_over_using_(-1),
    overuse_counter_(0),
    D_hypothesis_('N'),

    min_configured_bitrate_bps_(m_minBw), //initial value need
    max_configured_bitrate_bps_(m_maxBw),
    current_bitrate_bps_(min_configured_bitrate_bps_),
    latest_incoming_bitrate_bps_(current_bitrate_bps_),
    avg_max_bitrate_kbps_(-1.0f),
    var_max_bitrate_kbps_(0.4f),
    rate_control_state_('H'),
    rate_control_region_('M'),
    time_last_bitrate_change_(-1),
    time_first_incoming_estimate_(-1),
    bitrate_is_initialized_(false),
    beta_(kDefaultBackoffFactor),
    rtt_(kDefaultRttMs),
    in_experiment_(false),  // need initial value
    smoothing_experiment_(false),
    last_decrease_(0),

	  lost_packets_since_last_loss_update_(0),
    expected_packets_since_last_loss_update_(0),
	  min_bitrate_configured_(min_configured_bitrate_bps_),	
	  max_bitrate_configured_(max_configured_bitrate_bps_),    
    last_low_bitrate_log_ms_(-1),
    has_decreased_since_last_fraction_loss_(false),
    last_feedback_ms_(-1),
    last_packet_report_ms_(-1),
    last_timeout_ms_(-1),
    // last_fraction_loss_(0),  // this has been moved to SenderBasedController
    last_logged_fraction_loss_(0),
    last_round_trip_time_ms_(0),
    delay_based_bitrate_bps_(0),
    time_last_decrease_ms_(0),
    first_report_time_ms_(-1),
    initially_lost_packets_(0),
    bitrate_at_2_seconds_kbps_(0),
    in_timeout_experiment_(false),
    low_loss_threshold_(kDefaultLowLossThreshold),
    high_loss_threshold_(kDefaultHighLossThreshold),
    bitrate_threshold_bps_(1000 * kDefaultBitrateThresholdKbps) 

{	E_[0][0] = 100;
	E_[1][1] = 1e-1;
	E_[0][1] = E_[1][0] = 0;
	process_noise_[0] = 1e-10;
	process_noise_[1] = 1e-3;
}

GccController::~GccController() {}


void GccController::SetBitrates(int send_bitrate, int min_bitrate, int max_bitrate, int nowms) {
  SetMinMaxBitrate(min_bitrate, max_bitrate);
  if (send_bitrate > 0)
    SetSendBitrate(send_bitrate, nowms);
}

void GccController::SetSendBitrate(int bitrate, int nowms) {
  delay_based_bitrate_bps_ = 0;  // Reset to avoid being capped by the estimate.
  CapBitrateToThresholds(nowms, bitrate);	
  // Clear last sent bitrate history so the new value can be used directly
  // and not capped.
  min_bitrate_history_.clear();
}

void GccController::SetMinMaxBitrate(int min_bitrate, int max_bitrate) {
  min_bitrate_configured_ = std::max(min_bitrate, GetMinBitrate());
  if (max_bitrate > 0) {
    max_bitrate_configured_ = std::max<uint32_t>(min_bitrate_configured_, max_bitrate);
  } else {
    max_bitrate_configured_ = kDefaultMaxBitrateBps;
  }
}

int GccController::GetMinBitrate() const {
  return min_bitrate_configured_;
}

void GccController::UpdatePacketsLost(int packets_lost, int number_of_packets, int64_t now_ms) {
  last_feedback_ms_ = now_ms;
  if (first_report_time_ms_ == -1)
    first_report_time_ms_ = now_ms;

  // Check sequence number diff and weight loss report
  if (number_of_packets > 0) {
    // Accumulate reports.
    lost_packets_since_last_loss_update_ = packets_lost;
    expected_packets_since_last_loss_update_ = number_of_packets;
    // expected_packets_since_last_loss_update_ += packets_lost+1;

    // Don't generate a loss rate until it can be based on enough packets.
    if (expected_packets_since_last_loss_update_ < kLimitNumPackets)
      return;

    has_decreased_since_last_fraction_loss_ = false;
    int64_t lost_q8 = lost_packets_since_last_loss_update_ << 8;
    int64_t expected = expected_packets_since_last_loss_update_;
    last_fraction_loss_ = std::min<int>(lost_q8 / expected, 255);
	
    // Reset accumulators.

    lost_packets_since_last_loss_update_ = 0;
    expected_packets_since_last_loss_update_ = 0;
    last_packet_report_ms_ = now_ms;
    UpdateEstimate(now_ms);
  }
}



void GccController::CurrentEstimate(int* bitrate,
                                    uint8_t* loss,
                                    int64_t* rtt) const {
  *bitrate = current_bitrate_bps_;
  *loss = last_fraction_loss_;
  *rtt = last_round_trip_time_ms_;
}

void GccController::UpdateDelayBasedEstimate(
    int64_t now_ms,
    uint32_t bitrate_bps) {
  delay_based_bitrate_bps_ = bitrate_bps;
  CapBitrateToThresholds(now_ms, current_bitrate_bps_);
}

void GccController::UpdateEstimate(int64_t now_ms) {
  uint32_t new_bitrate = current_bitrate_bps_;
  // We trust the REMB and/or delay-based estimate during the first 2 seconds if
  // we haven't had any packet loss reported, to allow startup bitrate probing.
  if (last_fraction_loss_ == 0 && IsInStartPhase(now_ms)) {
    new_bitrate = std::max(delay_based_bitrate_bps_, new_bitrate);

    if (new_bitrate != current_bitrate_bps_) {
      min_bitrate_history_.clear();
      min_bitrate_history_.push_back(
          std::make_pair(now_ms, current_bitrate_bps_));
      CapBitrateToThresholds(now_ms, new_bitrate);
      return;
    }
  }
  UpdateMinHistory(now_ms);
  if (last_packet_report_ms_ == -1) {
    // No feedback received.
    CapBitrateToThresholds(now_ms, current_bitrate_bps_);
    return;
  }
  //int64_t time_since_packet_report_ms = now_ms - last_packet_report_ms_;
  //int64_t time_since_feedback_ms = now_ms - last_feedback_ms_;
  
  //  if (time_since_packet_report_ms < 1.2 * kFeedbackIntervalMs) {
  // We only care about loss above a given bitrate threshold.
  //float loss = last_fraction_loss_ / 256.0f;
  float loss = last_fraction_loss_ / 256.0f;
     
  // We only make decisions based on loss when the bitrate is above a
  // threshold. This is a crude way of handling loss which is uncorrelated
  // to congestion.
  if (current_bitrate_bps_ < bitrate_threshold_bps_ ||
      loss <= low_loss_threshold_) {
    // Loss < 2%: Increase rate by 8% of the min bitrate in the last
    // kBweIncreaseIntervalMs.
    // Note that by remembering the bitrate over the last second one can
    // rampup up one second faster than if only allowed to start ramping
    // at 8% per second rate now. E.g.:
    //   If sending a constant 100kbps it can rampup immediatly to 108kbps
    //   whenever a receiver report is received with lower packet loss.
    //   If instead one would do: current_bitrate_bps_ *= 1.08^(delta time),
    //   it would take over one second since the lower packet loss to achieve
    //   108kbps.
    new_bitrate = static_cast<uint32_t>(min_bitrate_history_.front().second * 1.08 + 0.5);
    // Add 1 kbps extra, just to make sure that we do not get stuck
    // (gives a little extra increase at low rates, negligible at higher
    // rates).
    new_bitrate += 1000;
    } else if (current_bitrate_bps_ > bitrate_threshold_bps_) {
      if (loss <= high_loss_threshold_) {
        // Loss between 2% - 10%: Do nothing.
      } else {
        // Loss > 10%: Limit the rate decreases to once a kBweDecreaseIntervalMs
        // + rtt.
        if (!has_decreased_since_last_fraction_loss_ &&
            (now_ms - time_last_decrease_ms_) >=
                (kBweDecreaseIntervalMs + last_round_trip_time_ms_)) {
          time_last_decrease_ms_ = now_ms;

          // Reduce rate:
          //   newRate = rate * (1 - 0.5*lossRate);
          //   where packetLoss = 256*lossRate;
          new_bitrate = static_cast<uint32_t>(
              (current_bitrate_bps_ *
               static_cast<double>(512 - last_fraction_loss_)) /
              512.0);
          has_decreased_since_last_fraction_loss_ = true;
        }
      }
    }
    /*} else if (time_since_feedback_ms >
                 kFeedbackTimeoutIntervals * kFeedbackIntervalMs &&
             (last_timeout_ms_ == -1 ||
              now_ms - last_timeout_ms_ > kTimeoutIntervalMs)) {
    if (in_timeout_experiment_) {
      new_bitrate *= 0.8;
      // Reset accumulators since we've already acted on missing feedback and
      // shouldn't to act again on these old lost packets.
      lost_packets_since_last_loss_update_ = 0;
      expected_packets_since_last_loss_update_ = 0;
      last_timeout_ms_ = now_ms;
    }
  }*/

  CapBitrateToThresholds(now_ms, new_bitrate);
}

bool GccController::IsInStartPhase(int64_t now_ms) const {
  return first_report_time_ms_ == -1 ||
         now_ms - first_report_time_ms_ < kStartPhaseMs;
}

void GccController::UpdateMinHistory(int64_t now_ms) {
  // Remove old data points from history.
  // Since history precision is in ms, add one so it is able to increase
  // bitrate if it is off by as little as 0.5ms.
  while (!min_bitrate_history_.empty() &&
         now_ms - min_bitrate_history_.front().first + 1 >
             kBweIncreaseIntervalMs) {
    min_bitrate_history_.pop_front();
  }

  // Typical minimum sliding-window algorithm: Pop values higher than current
  // bitrate before pushing it.

  while (!min_bitrate_history_.empty() &&
         current_bitrate_bps_ <= min_bitrate_history_.back().second) {
    min_bitrate_history_.pop_back();
  }

  min_bitrate_history_.push_back(std::make_pair(now_ms, current_bitrate_bps_));
}

void GccController::CapBitrateToThresholds(int64_t now_ms,
                                           uint32_t bitrate_bps) {

  if (delay_based_bitrate_bps_ > 0 && bitrate_bps > delay_based_bitrate_bps_ ) {
    bitrate_bps = delay_based_bitrate_bps_;
  }
 
  if (bitrate_bps > max_bitrate_configured_) {
    bitrate_bps = max_bitrate_configured_;
  }

  if (bitrate_bps < min_bitrate_configured_) {
    if (last_low_bitrate_log_ms_ == -1 ||
       now_ms - last_low_bitrate_log_ms_ > kLowBitrateLogPeriodMs) {
      last_low_bitrate_log_ms_ = now_ms;
    }
    bitrate_bps = min_bitrate_configured_;
  }

  if (bitrate_bps != current_bitrate_bps_ ||
      last_fraction_loss_ != last_logged_fraction_loss_ ||
      now_ms - last_rtc_event_log_ms_ > kRtcEventLogPeriodMs) {
    last_logged_fraction_loss_ = last_fraction_loss_;
    last_rtc_event_log_ms_ = now_ms;
  }
  current_bitrate_bps_ = bitrate_bps;
}



void GccController::setCurrentBw(float newBw) {
  m_initBw = newBw;
}

void GccController::reset() {
  m_lastTimeCalcUs = 0;
  m_lastTimeCalcValid = false;
	
  m_QdelayUs = 0;
  m_ploss = 0;
  m_Pkt = 0;
  m_plr = 0.f;
  m_RecvR = 0.;

  SenderBasedController::reset();
}

bool GccController::processFeedback(uint64_t nowUs,
                                    uint16_t sequence,
                                    uint64_t rxTimestampUs,
                                    int64_t l_inter_arrival,
                                    uint64_t l_inter_departure,
                                    int64_t l_inter_delay_var,
									                  int l_inter_group_size,	
									                  int64_t l_arrival_time,                                     
									                  uint8_t ecn) {
    // First of all, call the superclass
  const bool res = SenderBasedController::processFeedback(nowUs, sequence,
                                                          rxTimestampUs,
                                                          l_inter_arrival,
                                                          l_inter_departure,
                                                          l_inter_delay_var, l_inter_group_size, l_arrival_time,  ecn);		
	static const int kMinBitrateBps = min_bitrate_configured_;
	static const int kMaxBitrateBps = max_bitrate_configured_;
  static const int kInitialBitrateBps = min_bitrate_configured_;

	uint64_t now_ms = ns3::Simulator::Now().GetMilliSeconds();

	if(prev_seq_loss == 0){
		//initialize it
		prev_seq_loss = sequence ;
		m_timer = now_ms;
	}else if(m_timer + LOSS_TIMER < now_ms){			
		int delta_seq = sequence - prev_seq_loss;
		
		if(delta_seq != 0){
			float loss_ratio = (float)loss_counter / (float)delta_seq;
			loss_moving_avg = (loss_moving_avg * 0.8) + (loss_ratio * 0.2);			//EWMA	
			// std::cout << "loss rate every " << LOSS_TIMER << "ms : " << loss_ratio << ", moving_avg : " << loss_moving_avg << std::endl;
		}
		loss_counter = 0;
		prev_seq_loss = sequence + 1;
		m_timer = now_ms;
	}

	if(!res) return false;

	if(!m_lastTimeCalcValid){
	  m_lastTimeCalcValid = true;
		SetMinMaxBitrate(kMinBitrateBps, kMaxBitrateBps);
	 	SetSendBitrate(kInitialBitrateBps, nowUs);
		return true;
  }

	// Shift up send time to use the full 32 bits that inter_arrival works with,
  // so wrapping works properly.
  // TODO(holmer): SSRCs are only needed for REMB, should be broken out from
  // here.
  // Check if incoming bitrate estimate is valid, and if it needs to be reset.
 	updateMetrics();

  uint32_t ts_delta = (uint32_t) l_inter_departure/1000;
 	int64_t t_delta = l_inter_arrival/1000;
 	int size_delta = l_inter_group_size;
	
	bool update_estimate = false;	

	OveruseEstimatorUpdate(t_delta, ts_delta, size_delta, D_hypothesis_, l_arrival_time/1000);
  OveruseDetectorDetect(offset_, ts_delta, num_of_deltas_, l_arrival_time/1000);

  uint64_t rtt_us;
  getCurrentRTT(rtt_us);
  rtt_ = rtt_us / 1000;
  if (!update_estimate) {
    // Check if it's time for a periodic update or if we should update because
    // of an over-use.
            
		if (last_update_ms_ == -1 || (uint64_t)now_ms - last_update_ms_ > 1000){
   		update_estimate = true;
   	} 
    else if (D_hypothesis_ == 'O') {
      //update_estimate = true;//always reduce bitrate immediately
      uint32_t incoming_rate = std::min((uint32_t)m_RecvR, current_bitrate_bps_); 
			if (incoming_rate && TimeToReduceFurther(now_ms, incoming_rate)) {
        update_estimate = true;
      }
    }
  }

  // The first overuse should immediately trigger a new estimate.
  // We also have to update the estimate immediately if we are overusing
  // and the target bitrate is too high compared to what we are receiving.	
	
	//update_estimate = true;
	
	if (update_estimate){	
		Update(D_hypothesis_, std::min((uint32_t)m_RecvR, current_bitrate_bps_), var_noise_, now_ms);
		last_update_ms_ = now_ms;
    logStats(now_ms*1000);
 	}
  
  /*
  std::ofstream f_t("results/time_update.txt", std::ios::app);
  std::ofstream f_d("results/interdep.txt", std::ios::app);
  std::ofstream f_a("results/interarr.txt", std::ios::app);
  if(!f_t || !f_d || !f_a){
    std::cout<<"File open error"<<std::endl;
  }
  else{
    float time_s = float(now_ms / 1000);
    float interdep = float(ts_delta);
    float interarr = float(t_delta);
    f_t << time_s << std::endl;
    f_d << interdep << std::endl;
    f_a << interarr << std::endl;
    f_t.close();
    f_d.close();
    f_a.close();
  }*/

	UpdateDelayBasedEstimate(now_ms, current_bitrate_bps_);
  UpdateEstimate(now_ms);
  // UpdatePacketsLost(m_ploss, m_Pkt, now_ms);
	return res;
}

float GccController::getBandwidth(uint64_t nowUs) const {
    return m_initBw;
}

uint32_t GccController::getSendBps() const{
	return current_bitrate_bps_;
}

void GccController::updateMetrics() {
  uint64_t qdelayUs;
  bool qdelayOK = getCurrentQdelay(qdelayUs);
  if (qdelayOK) m_QdelayUs = qdelayUs;

  float rrate;
  bool rrateOK = getCurrentRecvRate(rrate);
  if (rrateOK) m_RecvR = rrate;
	
  uint32_t nLoss, nPkt;
  float plr;
  bool plrOK = getPktLossInfo(nLoss, plr, nPkt);
  if (plrOK) {
    m_ploss = nLoss;
    m_Pkt = nPkt;
    m_plr = plr;
  }	
}


void GccController::SetStartBitrate(int start_bitrate_bps) {
  current_bitrate_bps_ = start_bitrate_bps;
  latest_incoming_bitrate_bps_ = current_bitrate_bps_;
  bitrate_is_initialized_ = true;
}

void GccController::SetMinBitrate(int min_bitrate_bps) {
  min_configured_bitrate_bps_ = min_bitrate_bps;
  current_bitrate_bps_ = std::max<int>(min_bitrate_bps, current_bitrate_bps_);
}

bool GccController::ValidEstimate() const {
  return bitrate_is_initialized_;
}

uint64_t GccController::GetFeedbackInterval() const {
  // Estimate how often we can send RTCP if we allocate up to 5% of bandwidth
  // to feedback.
  static const int kRtcpSize = 80;
  const int64_t interval = static_cast<int64_t>(
      kRtcpSize * 8.0 * 1000.0 / (0.05 * current_bitrate_bps_) + 0.5);
  const int64_t kMinFeedbackIntervalMs = 200;
  return rtc::SafeClamp(interval, kMinFeedbackIntervalMs,
                        kMaxFeedbackIntervalMs);
}

bool GccController::TimeToReduceFurther(int32_t time_now,
                                        uint32_t incoming_bitrate_bps) const {
  const int64_t bitrate_reduction_interval =
      std::max<int64_t>(std::min<int64_t>(rtt_, 200), 10);
  if (time_now - time_last_bitrate_change_ >= bitrate_reduction_interval) {
    return true;
  }
  if (ValidEstimate()) {
    // TODO(terelius/holmer): Investigate consequences of increasing
    // the threshold to 0.95 * LatestEstimate().
    const uint32_t threshold = static_cast<uint32_t>(0.5 * LatestEstimate());
    return incoming_bitrate_bps < threshold;
  }
  return false;
}

uint32_t GccController::LatestEstimate() const {
  return current_bitrate_bps_;
}

uint32_t GccController::Update(char bw_state, uint32_t incoming_bitrate, double noise_var,
                                 int64_t now_ms) {
  // Set the initial bit rate value to what we're receiving the first half
  // second.
  // TODO(bugs.webrtc.org/9379): The comment above doesn't match to the code.
  if (!bitrate_is_initialized_) {
    const int64_t kInitializationTimeMs = 5000;
    
    if (time_first_incoming_estimate_ < 0) {
      if (incoming_bitrate)
        time_first_incoming_estimate_ = now_ms;
    }
    else if (now_ms - time_first_incoming_estimate_ > kInitializationTimeMs &&
             incoming_bitrate > 0) {
      current_bitrate_bps_ = incoming_bitrate;
      bitrate_is_initialized_ = true;
    }
  }

  current_bitrate_bps_ = ChangeBitrate(current_bitrate_bps_, bw_state, incoming_bitrate, noise_var, now_ms);
  bitrate_is_initialized_ = true;
  return current_bitrate_bps_;
}

void GccController::SetEstimate(int bitrate_bps, int64_t now_ms) {
  bitrate_is_initialized_ = true;
  current_bitrate_bps_ = ClampBitrate(bitrate_bps, bitrate_bps);
  time_last_bitrate_change_ = now_ms;
}

int GccController::GetNearMaxIncreaseRateBps() const {
  double bits_per_frame = static_cast<double>(current_bitrate_bps_) / 30.0;
  double packets_per_frame = std::ceil(bits_per_frame / (8.0 * 1200.0));
  double avg_packet_size_bits = bits_per_frame / packets_per_frame;

  // Approximate the over-use estimator delay to 100 ms.
  const int64_t response_time = in_experiment_ ? (rtt_ + 100) * 2 : rtt_ + 100;
  constexpr double kMinIncreaseRateBps = 4000;
  return static_cast<int>(std::max(
      kMinIncreaseRateBps, (avg_packet_size_bits * 1000) / response_time));
}

int GccController::GetExpectedBandwidthPeriodMs() const {
  const int kMinPeriodMs = smoothing_experiment_ ? 500 : 2000;
  constexpr int kDefaultPeriodMs = 3000;
  constexpr int kMaxPeriodMs = 50000;

  int increase_rate = GetNearMaxIncreaseRateBps();
  if (!last_decrease_)
    return smoothing_experiment_ ? kMinPeriodMs : kDefaultPeriodMs;

  return std::min(kMaxPeriodMs,
                  std::max<int>(1000 * static_cast<int64_t>(last_decrease_) /
                                    increase_rate,
                                kMinPeriodMs));
}

uint32_t GccController::ChangeBitrate(uint32_t new_bitrate_bps,
                                      char bw_state, uint32_t incoming_bitrate, double noise_var,
                                      int64_t now_ms) {
  uint32_t incoming_bitrate_bps = incoming_bitrate;
  if (incoming_bitrate)
    latest_incoming_bitrate_bps_ = incoming_bitrate;
  // An over-use should always trigger us to reduce the bitrate, even though
  // we have not yet established our first estimate. By acting on the over-use,
  // we will end up with a valid estimate.
  if (!bitrate_is_initialized_ &&
      bw_state == 'O')
    return current_bitrate_bps_;

  ChangeState(bw_state, now_ms);
  // if(now_ms>=124500 && now_ms<=128500){
  //       std::cout<<"time:"<<now_ms<<", region"<<bw_state<<", state"<<rate_control_state_<<std::endl;
  // }
  // Calculated here because it's used in multiple places.
  const float incoming_bitrate_kbps = incoming_bitrate_bps / 1000.0f;
  // Calculate the max bit rate std dev given the normalized
  // variance and the current incoming bit rate.
  const float std_max_bit_rate =
      sqrt(var_max_bitrate_kbps_ * avg_max_bitrate_kbps_);
  switch (rate_control_state_) {
    case 'H':
      break;

    case 'I':
      if (avg_max_bitrate_kbps_ >= 0 &&
          incoming_bitrate_kbps >
              avg_max_bitrate_kbps_ + 3 * std_max_bit_rate) {
        ChangeRegion('M');
        avg_max_bitrate_kbps_ = -1.0;
      }
      if (rate_control_region_ == 'N') {
        uint32_t additive_increase_bps =
            AdditiveRateIncrease(now_ms, time_last_bitrate_change_);
        new_bitrate_bps += additive_increase_bps;
      } else {
        uint32_t multiplicative_increase_bps = MultiplicativeRateIncrease(
            now_ms, time_last_bitrate_change_, new_bitrate_bps);
        new_bitrate_bps += multiplicative_increase_bps;
      }

      time_last_bitrate_change_ = now_ms;
      break;

    case 'D':
      // Set bit rate to something slightly lower than max
      // to get rid of any self-induced delay.
      new_bitrate_bps =
          static_cast<uint32_t>(beta_ * incoming_bitrate_bps + 0.5);
      if (new_bitrate_bps > current_bitrate_bps_) {
        // Avoid increasing the rate when over-using.
        if (rate_control_region_ != 'M') {
          new_bitrate_bps = static_cast<uint32_t>(
              beta_ * avg_max_bitrate_kbps_ * 1000 + 0.5f);
        }
        new_bitrate_bps = std::min(new_bitrate_bps, current_bitrate_bps_);
      }
      ChangeRegion('N');

      if (bitrate_is_initialized_ &&
          incoming_bitrate_bps < current_bitrate_bps_) {
        constexpr float kDegradationFactor = 0.9f;
        if (smoothing_experiment_ &&
            new_bitrate_bps <
                kDegradationFactor * beta_ * current_bitrate_bps_) {
          // If bitrate decreases more than a normal back off after overuse, it
          // indicates a real network degradation. We do not let such a decrease
          // to determine the bandwidth estimation period.
          last_decrease_ = 0;
        } else {
          last_decrease_ = current_bitrate_bps_ - new_bitrate_bps;
        }
      }
      if (incoming_bitrate_kbps <
          avg_max_bitrate_kbps_ - 3 * std_max_bit_rate) {
        avg_max_bitrate_kbps_ = -1.0f;
      }

      bitrate_is_initialized_ = true;
      UpdateMaxBitRateEstimate(incoming_bitrate_kbps);
      // Stay on hold until the pipes are cleared.
      rate_control_state_ = 'H';
      time_last_bitrate_change_ = now_ms;
      break;

    default:
      assert(false);
  }
  return ClampBitrate(new_bitrate_bps, incoming_bitrate_bps);
}

uint32_t GccController::ClampBitrate(uint32_t new_bitrate_bps,
                                     uint32_t incoming_bitrate_bps) const {
  // Don't change the bit rate if the send side is too far off.
  // We allow a bit more lag at very low rates to not too easily get stuck if
  // the encoder produces uneven outputs.
  const uint32_t max_bitrate_bps =
      static_cast<uint32_t>(1.5f * incoming_bitrate_bps) + 10000;
  if (new_bitrate_bps > current_bitrate_bps_ &&
      new_bitrate_bps > max_bitrate_bps && max_bitrate_bps > current_bitrate_bps_) {
    new_bitrate_bps = std::max(current_bitrate_bps_, max_bitrate_bps);
  }
  new_bitrate_bps = std::max(new_bitrate_bps, min_configured_bitrate_bps_);
  return new_bitrate_bps;
}

uint32_t GccController::MultiplicativeRateIncrease(
    int64_t now_ms,
    int64_t last_ms,
    uint32_t current_bitrate_bps) const {
  double alpha = 1.08;
  if (last_ms > -1) {
    auto time_since_last_update_ms =
        rtc::SafeMin<int64_t>(now_ms - last_ms, 1000);
    alpha = pow(alpha, time_since_last_update_ms / 1000.0);
  }
  uint32_t multiplicative_increase_bps =
      std::max(current_bitrate_bps * (alpha - 1.0), 1000.0);
  return multiplicative_increase_bps;
}

uint32_t GccController::AdditiveRateIncrease(int64_t now_ms,
                                             int64_t last_ms) const {
  return static_cast<uint32_t>((now_ms - last_ms) *
                               GetNearMaxIncreaseRateBps() / 1000);
}

void GccController::UpdateMaxBitRateEstimate(float incoming_bitrate_kbps) {
  const float alpha = 0.05f;
  if (avg_max_bitrate_kbps_ == -1.0f) {
    avg_max_bitrate_kbps_ = incoming_bitrate_kbps;
  } else {
    avg_max_bitrate_kbps_ =
        (1 - alpha) * avg_max_bitrate_kbps_ + alpha * incoming_bitrate_kbps;
  }
  // Estimate the max bit rate variance and normalize the variance
  // with the average max bit rate.
  const float norm = std::max(avg_max_bitrate_kbps_, 1.0f);
  var_max_bitrate_kbps_ =
      (1 - alpha) * var_max_bitrate_kbps_ +
      alpha * (avg_max_bitrate_kbps_ - incoming_bitrate_kbps) *
          (avg_max_bitrate_kbps_ - incoming_bitrate_kbps) / norm;
  // 0.4 ~= 14 kbit/s at 500 kbit/s
  if (var_max_bitrate_kbps_ < 0.4f) {
    var_max_bitrate_kbps_ = 0.4f;
  }
  // 2.5f ~= 35 kbit/s at 500 kbit/s
  if (var_max_bitrate_kbps_ > 2.5f) {
    var_max_bitrate_kbps_ = 2.5f;
  }
}

void GccController::ChangeState(char bw_state,
                                  int64_t now_ms) {
  
   switch (bw_state) {
    case 'N':
      if (rate_control_state_ == 'H') {
        time_last_bitrate_change_ = now_ms;
        rate_control_state_ = 'I';
      }
      if (rate_control_state_ == 'D') {
       rate_control_state_ = 'H';
      }
      break;
    case 'O':
      if (rate_control_state_ != 'D') {
        rate_control_state_ = 'D';
      }
      break;
    case 'U':
      rate_control_state_ = 'H';
      break;
    default:
      assert(false);
  }
}

void GccController::ChangeRegion(char region) {
  rate_control_region_ = region;
}



void GccController::logStats(uint64_t nowUs) const {

    // std::ostringstream os;
    // os << std::fixed;
    // os.precision(RMCAT_LOG_PRINT_PRECISION);

    // os  << " algo:gcc " << m_id
    //     << " ts: "     << (nowUs / 1000)
    //     << " loglen: " << m_packetHistory.size()
    //     << " qdel: "   << (m_QdelayUs / 1000)
    //     << " ploss: "  << m_ploss
    //     << " plr: "    << m_plr
    //     << " rrate: "  << m_RecvR
    //     << " srate: "  << current_bitrate_bps_;
    // logMessage(os.str());
    /*
    std::ofstream f_t("results/Time.txt", std::ios::app);
    std::ofstream f_r("results/SendRate.txt", std::ios::app);
    if(!f_r || !f_t){
      std::cout<<"File open error"<<std::endl;
    }
    else{
      float time_s = float(nowUs / 1000000);
      float send_rate = float(current_bitrate_bps_) / 1000.;
      f_t << time_s <<std::endl;
      f_r << send_rate <<std::endl;
      f_r.close();
      f_t.close();
    }*/
    
    /*
    FILE * f_fl = fopen("FractionLoss.bat", "a+b");
    FILE * f_t = fopen("Time.bat", "a+b");
    FILE * f_r = fopen("SendRate.bat", "a+b");
    if(f_fl == NULL || f_t == NULL || f_r == NULL){
      std::cout<<"File open error!"<<std::endl;
    }
    else
    {   
      float time_s = float(nowUs / 1000000);
      float send_rate = float(current_bitrate_bps_) / 1000.;
      count += 1;
      fwrite(&(m_plr), sizeof(float), 1, f_fl);
      fclose(f_fl);
      fwrite(&(time_s), sizeof(float), 1, f_t);
      fclose(f_t);
      fwrite(&(send_rate), sizeof(float), 1, f_r);
      fclose(f_r);
      std::cout<<send_rate<<" count:"<<count<<std::endl;
    }*/
}


void GccController::OveruseEstimatorUpdate(int64_t t_delta, double ts_delta, int size_delta, char current_hypothesis, int64_t now_ms){
	// const double min_frame_period = UpdateMinFramePeriod(ts_delta);
	const double t_ts_delta = t_delta - ts_delta; // t_ts_delta is dm(t) in webrtc paper
	
	++num_of_deltas_;
	if (num_of_deltas_ > kDeltaCounterMax) {
   	num_of_deltas_ = kDeltaCounterMax;
  }

  accumulated_delay_ += t_ts_delta;

  smoothed_delay_ = smoothing_coef_ * smoothed_delay_ +
                      (1 - smoothing_coef_) * accumulated_delay_;

  // Simple linear regression.
  delay_hist_.push_back(std::make_pair(
      static_cast<double>(this->m_pkt_cnt++ * 16.67),
      smoothed_delay_));

  if(delay_hist_.size() > window_size_) {
      delay_hist_.pop_front();
  }

//   for(auto entry : delay_hist_) {
//       DEBUG("[GCC] At " << now_ms << " ms: " << entry.first << ", " << entry.second);
//   }
  
  delay_hist_.shrink_to_fit();

  if(delay_hist_.size() == window_size_) {
      // Update trend_ if it is possible to fit a line to the data. The delay
      // trend can be seen as an estimate of (send_rate - capacity)/capacity.
      // 0 < trend < 1   ->  the delay increases, queues are filling up
      //   trend == 0    ->  the delay does not change
      //   trend < 0     ->  the delay decreases, queues are being emptied
      offset_ = LinearFitSlope(delay_hist_);  // offset_ in kalkan is the trend in trendline
  }
	

  // Update the Kalman filter.
  /*
  E_[0][0] += process_noise_[0];
  E_[1][1] += process_noise_[1];

  if ((current_hypothesis == 'O' && offset_ < prev_offset_) ||
      (current_hypothesis == 'U' && offset_ > prev_offset_)) {
    E_[1][1] += 10 * process_noise_[1];
  }

  const double h[2] = {fs_delta, 1.0};
	
  const double Eh[2] = {E_[0][0] * h[0] + E_[0][1] * h[1],
                        E_[1][0] * h[0] + E_[1][1] * h[1]};

 	const double residual = t_ts_delta - slope_ * h[0] - offset_; // residual is z(t)

  const bool in_stable_state = (current_hypothesis == 'N');
  const double max_residual = 3.0 * sqrt(var_noise_);

  // We try to filter out very late frames. For instance periodic key
	// frames doesn't fit the Gaussian model well.
  if (fabs(residual) < max_residual) {
    UpdateNoiseEstimate(residual, min_frame_period, in_stable_state);
  } else {
    UpdateNoiseEstimate(residual < 0 ? -max_residual : max_residual,
                        min_frame_period, in_stable_state);
  }

  const double denom = var_noise_ + h[0] * Eh[0] + h[1] * Eh[1];

  const double K[2] = {Eh[0] / denom, Eh[1] / denom};

  const double IKh[2][2] = {{1.0 - K[0] * h[0], -K[0] * h[1]},
                            {-K[1] * h[0], 1.0 - K[1] * h[1]}};
  const double e00 = E_[0][0];
	const double e01 = E_[0][1];
                              
	// Update state.            
  E_[0][0] = e00 * IKh[0][0] + E_[1][0] * IKh[0][1];
  E_[0][1] = e01 * IKh[0][0] + E_[1][1] * IKh[0][1];
  E_[1][0] = e00 * IKh[1][0] + E_[1][0] * IKh[1][1];
  E_[1][1] = e01 * IKh[1][0] + E_[1][1] * IKh[1][1];
  
 	// The covariance matrix must be positive semi-definite.
  bool positive_semi_definite =
      E_[0][0] + E_[1][1] >= 0 &&
      		E_[0][0] * E_[1][1] - E_[0][1] * E_[1][0] >= 0 && E_[0][0] >= 0;
  	

  assert(positive_semi_definite);
  

  slope_ = slope_ + K[0] * residual;
  prev_offset_ = offset_;
  offset_ = offset_ + K[1] * residual; // offset_ is m(t)
  */
}

double GccController::LinearFitSlope(const std::deque<std::pair<double, double>>& points)
{
    // Compute the "center of mass".
    double sum_x = 0;
    double sum_y = 0;
    for(const std::pair<double, double>& point : points) {
        sum_x += point.first;
        sum_y += point.second;
    }
    double x_avg = sum_x / points.size();
    double y_avg = sum_y / points.size();
    // Compute the slope k = \sum (x_i-x_avg)(y_i-y_avg) / \sum (x_i-x_avg)^2
    double numerator = 0;
    double denominator = 0;
    for(const std::pair<double, double>& point : points) {
        numerator += (point.first - x_avg) * (point.second - y_avg);
        denominator += (point.first - x_avg) * (point.first - x_avg);
    }
    if(denominator == 0) {
        return 0;
    }
    return numerator / denominator;
}

double GccController::UpdateMinFramePeriod(double ts_delta) {
  double min_frame_period = ts_delta;
  if (ts_delta_hist_.size() >= kMinFramePeriodHistoryLength) {
    ts_delta_hist_.pop_front();
  }
  for (const double old_ts_delta : ts_delta_hist_) {
    min_frame_period = std::min(old_ts_delta, min_frame_period);
  }
  ts_delta_hist_.push_back(ts_delta);
  return min_frame_period;
}

void GccController::UpdateNoiseEstimate(double residual,
                                           double ts_delta,
                                           bool stable_state) {
  if (!stable_state) {
    return;
  }
  // Faster filter during startup to faster adapt to the jitter level
  // of the network. |alpha| is tuned for 30 frames per second, but is scaled
  // according to |ts_delta|.
  double alpha = 0.01;
  if (num_of_deltas_ > 10 * 30) {
    alpha = 0.002;
  }
  // Only update the noise estimate if we're not over-using. |beta| is a
  // function of alpha and the time delta since the previous update.
  const double beta = pow(1 - alpha, ts_delta * 30.0 / 1000.0);
  avg_noise_ = beta * avg_noise_ + (1 - beta) * residual;
  var_noise_ = beta * var_noise_ +
               (1 - beta) * (avg_noise_ - residual) * (avg_noise_ - residual);
  if (var_noise_ < 1) {
    var_noise_ = 1;
  }
}

void GccController::UpdateThreshold(double modified_offset, int64_t now_ms){

  if (ut_last_update_ms_ == -1)
    ut_last_update_ms_ = now_ms;

  if (fabs(modified_offset) > threshold_ + kMaxAdaptOffsetMs) {
    // Avoid adapting the threshold to big latency spikes, caused e.g.,
    // by a sudden capacity drop.
    ut_last_update_ms_ = now_ms;
    return;
  }

  const double k = fabs(modified_offset) < threshold_ ? k_down_ : k_up_;
  const int64_t kMaxTimeDeltaMs = 100;
  int64_t time_delta_ms = std::min(now_ms - ut_last_update_ms_, kMaxTimeDeltaMs);
  threshold_ += k * (fabs(modified_offset) - threshold_) * time_delta_ms;
  threshold_ = rtc::SafeClamp(threshold_, 6.f, 600.f);
  ut_last_update_ms_ = now_ms;
	
}

char GccController::State() const {
	return D_hypothesis_;
}

char GccController::OveruseDetectorDetect(double offset, double ts_delta, int num_of_deltas, int64_t now_ms){

  if (num_of_deltas < 2) {
    return 'N';
  }

	
  //const double T = std::min(num_of_deltas, kMinNumDeltas) * offset;
  const double T = std::min(num_of_deltas, kMinNumDeltas) * offset * threshold_gain_;
  if (T > threshold_) {
    if (time_over_using_ == -1) {
      // Initialize the timer. Assume that we've been
      // over-using half of the time since the previous
      // sample.
      time_over_using_ = ts_delta / 2;
    } else {
      // Increment timer
      time_over_using_ += ts_delta;
    }
    overuse_counter_++;

    if (time_over_using_ > overusing_time_threshold_ && overuse_counter_ > 1) {
      if (offset >= D_prev_offset_) {
        time_over_using_ = 0;
        overuse_counter_ = 0;
        D_hypothesis_ = 'O';
      }
    }
  } else if (T < -threshold_) {
    time_over_using_ = -1;
    overuse_counter_ = 0;
    D_hypothesis_ = 'U';
  } else {
    time_over_using_ = -1;
    overuse_counter_ = 0;
    D_hypothesis_ = 'N';
	
  }
  D_prev_offset_ = offset;


  UpdateThreshold(T, now_ms);

  /*
  std::ofstream f_t("results/time_m.txt", std::ios::app);
  std::ofstream f_m("results/mt.txt", std::ios::app);
  std::ofstream f_g("results/gamma.txt", std::ios::app);
  if(!f_t || !f_m || !f_g){
    std::cout<<"File open error"<<std::endl;
  }
  else{
    float time_s = float(now_ms) / 1000.;
   float mt = float(T);
    f_t << time_s << std::endl;
    f_m << mt << std::endl;
    f_g << threshold_ << std::endl; 
    f_t.close();
    f_m.close();
    f_g.close();
  }*/
  

  return D_hypothesis_;

}

} // namespace rmcat
