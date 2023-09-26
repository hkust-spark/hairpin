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
 * Dummy controller (CBR) interface for rmcat ns3 module.
 *
 * @version 0.1.1
 * @author Jiantao Fu
 * @author Sergio Mena
 * @author Xiaoqing Zhu
 */

#ifndef GCC_CONTROLLER_H
#define GCC_CONTROLLER_H

#include "sender-based-controller.h"
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

#include "ns3/checks.h"
#include "ns3/safe_minmax.h"



namespace rmcat {

/**
 * Simplistic implementation of a sender-based congestion controller. The
 * algorithm simply returns a constant, hard-coded bandwidth when queried.
 */
class GccController: public SenderBasedController
{
public:
    /** Class constructor */
    GccController();

    /** Class destructor */
    virtual ~GccController();

    /**
     * Set the current bandwidth estimation. This can be useful in test environments
     * to temporarily disrupt the current bandwidth estimation
     *
     * @param [in] newBw Bandwidth estimation to overwrite the current estimation
     */
    virtual void setCurrentBw(float newBw);

    /**
     * Reset the internal state of the congestion controller
     */
    virtual void reset();

    /**
     * Simplistic implementation of feedback packet processing. It simply
     * prints calculated metrics at regular intervals
     */
    virtual bool processFeedback(uint64_t nowUs,
                                 uint16_t sequence,
                                 uint64_t rxTimestampUs,
                                 int64_t l_inter_arrival,
                                 uint64_t l_inter_departure,
                                 int64_t l_inter_delay_var,
								 int l_inter_group_size,
								 int64_t l_arrival_time,
                                 uint8_t ecn=0);
    /**
     * Simplistic implementation of bandwidth getter. It returns a hard-coded
     * bandwidth value in bits per second
     */
    virtual float getBandwidth(uint64_t nowUs) const;

	virtual uint32_t getSendBps() const;	

/*Overuse Estimator Function */
    void OveruseEstimatorUpdate(int64_t t_delta, double ts_delta, int size_delta, char current_hypothesis, int64_t now_ms);
	
/*Overuse Detector Function */
    char OveruseDetectorDetect(double offset, double timestamp_delta, int num_of_deltas, int64_t now_ms);
    char State() const;

/*Delay Based Rate Controller Function*/
    bool ValidEstimate() const;
    void SetStartBitrate(int start_bitrate_bps);
    void SetMinBitrate(int min_bitrate_bps);
    uint64_t GetFeedbackInterval() const;

    bool TimeToReduceFurther(int32_t time_now, uint32_t incoming_bitrate_bps) const;

    uint32_t LatestEstimate() const;
    void SetRtt(int64_t rtt);
    uint32_t Update(char bw_state, uint32_t incoming_bitrate, double noise_var, int64_t now_ms);

    void SetEstimate(int bitrate_bps, int64_t now_ms);

    int GetNearMaxIncreaseRateBps() const;
    int GetExpectedBandwidthPeriodMs() const;

/*Loss Based Rate Controller Function */
    void CurrentEstimate(int* bitrate, uint8_t* loss, int64_t* rtt) const;

  	// Call periodically to update estimate.
  	void UpdateEstimate(int64_t now_ms);

  	// Call when a new delay-based estimate is available.
  	void UpdateDelayBasedEstimate(int64_t now_ms, uint32_t bitrate_bps);

  	void SetBitrates(int send_bitrate, int min_bitrate, int max_bitrate, int nowms);
  	void SetSendBitrate(int bitrate, int nowms);
  	void SetMinMaxBitrate(int min_bitrate, int max_bitrate);
  	int GetMinBitrate() const;
	void UpdatePacketsLost(int packet_lost, int number_of_packets, int64_t now_ms);

	


private:
/*Overuse Estimator Function */
    double LinearFitSlope(const std::deque<std::pair<double, double>>& points);
    void Detect(double trend, double ts_delta, int64_t now_ms);
    double UpdateMinFramePeriod(double ts_delta);
    void UpdateNoiseEstimate(double residual, double ts_delta, bool stable_state);

/*Overuse Detector Function */
    void UpdateThreshold(double modified_offset, int64_t now_ms);

/*Delay Based Rate Controller Function*/
    uint32_t ChangeBitrate(uint32_t current_bitrate, char bw_state, uint32_t incoming_bitrate, double noise_var, int64_t now_ms);
    uint32_t ClampBitrate(uint32_t new_bitrate_bps, uint32_t incoming_bitrate_bps) const;
    uint32_t MultiplicativeRateIncrease(int64_t now_ms, int64_t last_ms, uint32_t current_bitrate_bps) const;
    uint32_t AdditiveRateIncrease(int64_t now_ms, int64_t last_ms) const;
    void UpdateChangePeriod(int64_t now_ms);
    void UpdateMaxBitRateEstimate(float incoming_bit_rate_kbps);
    void ChangeState(char bw_state, int64_t now_ms);
    void ChangeRegion(char region);

    void updateMetrics();
    void logStats(uint64_t nowUs) const;

/*Loss Based Rate controller Function*/
  bool IsInStartPhase(int64_t now_ms) const;
  void UpdateMinHistory(int64_t now_ms);
  void CapBitrateToThresholds(int64_t now_ms, uint32_t bitrate_bps);

/* private variables */


    uint64_t m_pkt_cnt;

    uint64_t m_lastTimeCalcUs;
    bool m_lastTimeCalcValid;

    uint64_t m_QdelayUs; /**< estimated queuing delay in microseconds */
    uint32_t m_Pkt;
	uint32_t m_ploss;  /**< packet loss count within configured window */
    float m_plr;       /**< packet loss ratio within packet history window */
    float m_RecvR;     /**< updated receiving rate in bps */
	uint64_t m_timer;
	int prev_seq_loss;  // count loss ratio
	float loss_moving_avg;	
	float m_plrmoving_avg;
	
/*Overuse Estimator variable*/
    uint16_t num_of_deltas_;
    double slope_;
    double offset_;
    double prev_offset_;
    double E_[2][2];
    double process_noise_[2];
    double avg_noise_;
    double var_noise_;
    std::deque<double> ts_delta_hist_;
    
/* trendline estimator */
    const size_t window_size_;
    double smoothing_coef_;
    const double threshold_gain_;
    double accumulated_delay_;
    double smoothed_delay_;
        // Linear least squares regression.
    std::deque<std::pair<double, double>> delay_hist_;

/*Overuse Detector variable*/
    double k_up_;
    double k_down_;
    double overusing_time_threshold_;
    double threshold_;
    int64_t last_update_ms_;
    int64_t ut_last_update_ms_;
    double D_prev_offset_;
    double time_over_using_;
    int overuse_counter_;
    char D_hypothesis_; // O : Overusing, N : Normal, U : Underusing

/*Delay Based Rate Controller*/

    uint32_t min_configured_bitrate_bps_;
    uint32_t max_configured_bitrate_bps_;
    uint32_t current_bitrate_bps_;
    uint32_t latest_incoming_bitrate_bps_;
    float avg_max_bitrate_kbps_;
    float var_max_bitrate_kbps_;
    char rate_control_state_;   // H : Hold, I : Increase, D : Decrease
    char rate_control_region_;  // M : MaxUnkown, N : NearMax
    int64_t time_last_bitrate_change_;
    int64_t time_first_incoming_estimate_;
    bool bitrate_is_initialized_;
    float beta_;
    int64_t rtt_;
    bool in_experiment_;
    bool smoothing_experiment_;
    int last_decrease_;


   	std::deque<std::pair<int64_t, uint32_t> > min_bitrate_history_;

  	// incoming filters
  	int lost_packets_since_last_loss_update_;
  	int expected_packets_since_last_loss_update_;

  	uint32_t min_bitrate_configured_;
  	uint32_t max_bitrate_configured_;
  	int64_t last_low_bitrate_log_ms_;

  	bool has_decreased_since_last_fraction_loss_;
  	int64_t last_feedback_ms_;
  	int64_t last_packet_report_ms_;
  	int64_t last_timeout_ms_;
  	uint8_t last_fraction_loss_;
  	uint8_t last_logged_fraction_loss_;
  	int64_t last_round_trip_time_ms_;

  	uint32_t bwe_incoming_;
  	uint32_t delay_based_bitrate_bps_;
  	int64_t time_last_decrease_ms_;
  	int64_t first_report_time_ms_;
  	int initially_lost_packets_;
  	int bitrate_at_2_seconds_kbps_;
 	int64_t last_rtc_event_log_ms_;
  	bool in_timeout_experiment_;
  	float low_loss_threshold_;
  	float high_loss_threshold_;
  	uint32_t bitrate_threshold_bps_;

};

}

#endif /* DUMMY_CONTROLLER_H */
