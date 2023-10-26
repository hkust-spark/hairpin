#include "ns3/core-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/nstime.h"
#include "ns3/spark-rtc-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/common-header.h"

#include <memory>
#include <fstream>
#include <boost/filesystem.hpp>


using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("FECEmulator");

const uint32_t TOPO_DEFAULT_BW     = 30000;    // in Mbps: 30 Gbps
const uint32_t TOPO_DEFAULT_PDELAY = 10;     // in ms:   RTT: 20ms
const uint32_t TOPO_DEFAULT_QDELAY = 300;     // in ms:  300ms
const uint32_t DEFAULT_PACKET_SIZE = 1000;
const double   DEFAULT_ERROR_RATE  = 0.00;
const int      DEFAULT_NETWORK_CHANGE_INTERVAL = 16; //in ms


const uint8_t  FRAME_PER_SECOND = 60;  // 60 fps
const uint32_t BITRATE_MBPS     = 30;  // in Mbps: 30 Mbps
const uint8_t  DELAY_DDL_MS     = 100;

const uint32_t DEFAULT_RECEIVER_WINDOW = 34; // in ms

const float    EMULATION_DURATION = 300.; // in s

std::string DEFAULT_TRACE = "traces/cleaned-011117/153sid:1027.log";

// Network topology
//
// [GameServer]        [Error] [Rate Limit]  [GameClient]
//      ⬇                  ⬇  ⬇                 ⬇
//      n0 ----------------- n1 ----------------- n2
//   10.1.1.1             10.1.1.2             10.1.1.3


static NodeContainer BuildExampleTopo (uint64_t bps,
                                       double_t ms_delay,
                                       uint32_t msQdelay,
                                       std::string dir,
                                       bool isPcapEnabled)
{
  NS_LOG_INFO ("Create nodes.");
  NodeContainer c;
  c.Create (3);
  NodeContainer n0n1 = NodeContainer (c.Get (0), c.Get (1));
  NodeContainer n2n1 = NodeContainer (c.Get (2), c.Get (1));

  InternetStackHelper internet;
  internet.Install (c);

  NS_LOG_INFO ("Create channels.");
  PointToPointHelper p2p;
  /* Far beyond max bw, related to delay smoothness params */
  p2p.SetDeviceAttribute ("DataRate", DataRateValue (DataRate (1e9)));
  p2p.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (ms_delay - 1)));
  NetDeviceContainer d0d1 = p2p.Install (n0n1);

  DEBUG("bps: " << bps << ", ms_delay: " << ms_delay);
  // PointToPointHelper p2p2;
  p2p.SetDeviceAttribute ("DataRate", DataRateValue (DataRate (bps)));
  p2p.SetChannelAttribute ("Delay", StringValue("1ms"));
  p2p.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("1000p"));
  NetDeviceContainer d2d1 = p2p.Install (n2n1);

  NS_LOG_INFO ("Assign IP Addresses.");
  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  ipv4.Assign (d0d1);
  ipv4.SetBase ("10.1.2.0", "255.255.255.0");
  ipv4.Assign (d2d1);

  NS_LOG_INFO ("Use global routing.");
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  if (isPcapEnabled) {
    p2p.EnablePcapAll (dir + "/pcap");
    AsciiTraceHelper ascii;
    p2p.EnableAsciiAll (ascii.CreateFileStream (dir + "/pcap.tr"));
  }


  // disable tc for now, some bug in ns3 causes extra delay
  //TrafficControlHelper tch;
  //tch.Uninstall (devices);

	// add random loss
	std::string errorModelType = "ns3::RateErrorModel";
  ObjectFactory factory;
  factory.SetTypeId (errorModelType);
  Ptr<ErrorModel> em = factory.Create<ErrorModel> ();
	d0d1.Get(1)->SetAttribute ("ReceiveErrorModel", PointerValue (em)); /* Set error model for n1 */

  return c;
}

double_t oldDelay, newDelay;
double_t remainInterval;
std::streampos pos = std::ios::beg;
double_t minSetIntervalMs = 0.013;
double_t maxAllowedDiffMs = 0.011;


double_t appStart          = 0.;
double_t appStop;

void BandwidthTrace (Ptr<Node> node0, 
                     Ptr<Node> node1, 
                     std::string trace, 
                     bool bwChange, 
                     int interval,
                     bool readNewLine) {
  Ptr<PointToPointNetDevice> n0SndDev = StaticCast<PointToPointNetDevice,NetDevice> (node0->GetDevice (1));
  Ptr<PointToPointNetDevice> n1RcvDev = StaticCast<PointToPointNetDevice,NetDevice> (node1->GetDevice (1));
  Ptr<PointToPointNetDevice> n1SndDev = StaticCast<PointToPointNetDevice,NetDevice> (node1->GetDevice (2));
  std::ifstream traceFile;
  std::string newBwStr = "300Mbps";
  double newErrorRate = DEFAULT_ERROR_RATE;
  NS_ASSERT (maxAllowedDiffMs <= minSetIntervalMs);

  if (readNewLine) {
    std::string traceLine;
    std::vector<std::string> traceData;
    std::vector<std::string> bwValue;
    std::vector<std::string> rttValue;

    traceLine.clear ();
    traceFile.open (trace);
    traceFile.seekg (pos);
    std::getline (traceFile, traceLine);
    pos = traceFile.tellg ();
    if (traceLine.find (' ') == std::string::npos) {
      traceFile.close ();
      return;
    }
    rttValue.clear ();
    bwValue.clear ();
    traceData.clear ();
    SplitString (traceLine, traceData," ");
    SplitString (traceData[0], bwValue, "Mbps");
    SplitString (traceData[1], rttValue, "ms");
    
    /* Set delay of n0-n1 as rtt/2 - 1, the delay of n1-n2 is 1ms */ 
    newDelay = std::stod (rttValue[0]) / 2. - 1;
    newBwStr = std::to_string (std::stod (bwValue[0]) * 1.5) + "Mbps";
    newErrorRate = std::stod (traceData[2]);
    NS_LOG_FUNCTION (Simulator::Now ().GetMilliSeconds () << 
      " delay " << newDelay << " bw " << newBwStr << " errorRate " << newErrorRate);

    if (bwChange) {
      // Set bandwidth
      n1SndDev->SetAttribute ("DataRate", StringValue (newBwStr));
    }

    // Set error rate
    ObjectFactory factoryErrModel;
    factoryErrModel.SetTypeId ("ns3::RateErrorModel");
    factoryErrModel.Set ("ErrorUnit", EnumValue (RateErrorModel::ERROR_UNIT_PACKET),
                         "ErrorRate", DoubleValue (newErrorRate));
    Ptr<ErrorModel> em = factoryErrModel.Create<ErrorModel> ();
    n1RcvDev->SetAttribute ("ReceiveErrorModel", PointerValue (em));
    remainInterval = interval;
  }
  
  /* Set propagation delay, smoothsize to decrease 0.3ms every 0.33ms to avoid out of order
     These values are calculated based on 30Mbps and 1500B MTU --> 0.4ms / pkt */
  Ptr<Channel> channel = n0SndDev->GetChannel ();
  bool smoothDecrease = newDelay < oldDelay - maxAllowedDiffMs ? 1 : 0;
  oldDelay = smoothDecrease ? (oldDelay - maxAllowedDiffMs) : newDelay;
  channel->SetAttribute ("Delay", StringValue (std::to_string (oldDelay) + "ms"));
  if (smoothDecrease && remainInterval > minSetIntervalMs) {
    Simulator::Schedule (MicroSeconds (minSetIntervalMs * 1000), &BandwidthTrace, node0, node1, trace, 
      bwChange, interval, false);
    remainInterval -= minSetIntervalMs;
  } else {
    if ((!traceFile.eof ()) && Simulator::Now () < Seconds (appStop + 2)) {
      traceFile.close ();
      Simulator::Schedule (MicroSeconds (remainInterval * 1000), &BandwidthTrace, node0, node1, trace, 
        bwChange, interval, true);
    }
  } 
}

static void InstallApp (Ptr<Node> sender,
                        Ptr<Node> receiver,
                        uint16_t port,
                        float startTime_s,
                        float stopTime_s,
                        Ptr<FECPolicy> fecPolicy,
                        std::string rtxPolicy,
                        uint8_t fps,
                        uint16_t interval,
                        uint8_t delay_ddl_ms,
                        uint32_t bitrate_bps,
                        int ccOption,
                        uint16_t default_rtt /* in ms */,
                        double_t default_bw/* in Mbps */,
                        double_t default_loss,
                        uint32_t receiver_window/* in ms*/,
                        bool set_trace,
                        std::string tracefile,
                        std::string dir)
{
  AsciiTraceHelper ascii;
  std::string fecTrFileName = dir + "/fec.tr";
  std::string appTrFileName = dir + "/app.tr";
  std::string debugTrFileName = dir + "/debug.tr";
  Ptr<OutputStreamWrapper> fecStream = ascii.CreateFileStream (fecTrFileName);
  Ptr<OutputStreamWrapper> appStream = ascii.CreateFileStream (appTrFileName);
  Ptr<OutputStreamWrapper> debugStream = ascii.CreateFileStream (debugTrFileName);

  // Install applications
  Ptr<GameServer> sendApp = CreateObject<GameServer> ();
  Ptr<GameClient> recvApp = CreateObject<GameClient> ();

  Ptr<Ipv4> ipv4_recv = receiver->GetObject<Ipv4> ();
  Ipv4Address receiverIp = ipv4_recv->GetAddress (1, 0).GetLocal ();
  Ptr<Ipv4> ipv4_send = sender->GetObject<Ipv4> ();
  Ipv4Address senderIp = ipv4_send->GetAddress (1, 0).GetLocal ();

  sender->AddApplication (sendApp);
  receiver->AddApplication (recvApp);

  sendApp->Setup (
    senderIp, port, receiverIp, port,
    fps, MilliSeconds(delay_ddl_ms), bitrate_bps / 1e3, interval,
    fecPolicy, rtxPolicy, MilliSeconds (receiver_window),
    default_rtt, default_bw, default_loss, 0.01,
    set_trace, tracefile, fecStream, debugStream
  );
  recvApp->Setup (senderIp, port, port, fps, MicroSeconds(delay_ddl_ms * 1e3), receiver_window * 1000, default_rtt,
    appStream, debugStream);

  EventId stop_encoding_event = Simulator::Schedule(Seconds(stopTime_s), &GameServer::StopEncoding, sendApp);

  if (ccOption == GCC) {
    sendApp->SetController (GCC);
  } else if (ccOption == NADA) {
    sendApp->SetController (NADA);
  }

  sendApp->SetStartTime(Seconds(startTime_s));
  sendApp->SetStopTime(Seconds(stopTime_s + 2));
  recvApp->SetStartTime(Seconds(startTime_s));
  recvApp->SetStopTime(Seconds(stopTime_s + 2));
}

static void EnableLogging(LogLevel log_level) {
    LogComponentEnable("FECEmulator", LOG_LEVEL_INFO);

    LogComponentEnable("FECPolicy", log_level);
    LogComponentEnable("Hairpin", log_level);
    LogComponentEnable("OtherPolicies", log_level);

    LogComponentEnable("VideoEncoder", log_level);
    LogComponentEnable("VideoDecoder", log_level);
    LogComponentEnable("GameServer", log_level);
    LogComponentEnable("GameClient", log_level);
    LogComponentEnable("PacketSender", log_level);
    LogComponentEnable("PacketReceiver", log_level);

    LogComponentEnable("NetworkPacket", log_level);
    LogComponentEnable("PacketHeader", log_level);
    LogComponentEnable("PacketBatch", log_level);
}



int
main (int argc, char *argv[])
{

    // arguments
    std::string fecPolicy   = "fixed";
    std::string rtxPolicy   = "dupack";
    uint16_t  delayDdl      = DELAY_DDL_MS;
    double_t loss_rate      = DEFAULT_ERROR_RATE;
    bool fixed_loss_flag    = false; /* for static test */
    // bool fixed_rtt_flag     = false; /* Warning: DO NOT set RTT fixed!!! Fixed rtt causes measurement error when FECPolicy tries to calculate possibility */
    int      k_order        = 1;  /* for WebRTCStarPolicy */
    uint64_t linkBw         = TOPO_DEFAULT_BW;
    double_t ms_delay       = TOPO_DEFAULT_PDELAY;
    uint32_t msQDelay       = TOPO_DEFAULT_QDELAY;
    uint16_t fps            = FRAME_PER_SECOND;
    uint32_t bitrate        = BITRATE_MBPS;
    float duration          = EMULATION_DURATION;  // in s
    int cc_option           = NOT_USE_CC;
    bool network_variation  = false;
    bool set_trace          = false;
    uint16_t variation_interval  = DEFAULT_NETWORK_CHANGE_INTERVAL;
    uint32_t receiver_wnd   = DEFAULT_RECEIVER_WINDOW;
    double_t max_fec_rate   = -1; /* Bound FEC rate using this parameter */
    bool isPcapEnabled      = false;
    std::string logDir      = "logs";

    double_t qoeCoeff = 1e-7;

    int port = 8000;    /* application port */
    std::string trace = DEFAULT_TRACE;

    // fixed-hairpin-policy
    uint16_t fixed_group_size = 0;

    double_t param1 = 1;

    CommandLine cmd;
    cmd.AddValue("fecPolicy",   "FECPolicy, one of [hairpin, fixed, webrtc, awebrtc, webrtcstar, rtx, fec]", fecPolicy);
    cmd.AddValue("rtxPolicy",   "FECPolicy, one of [hairpin, fixed, webrtc, awebrtc, webrtcstar, rtx, fec]", rtxPolicy);
    cmd.AddValue("ddl",      "Frame deadline, in ms", delayDdl);
    cmd.AddValue("loss",     "Link packet loss rate, [0, 1]", loss_rate);
    cmd.AddValue("bw",       "Link bandwidth, in Mbps", linkBw);
    cmd.AddValue("delay",    "One-way delay, in ms", ms_delay);
    cmd.AddValue("fps",      "Frame per second", fps);
    cmd.AddValue("bitrate",  "Video encoding bitrate, in Mbps", bitrate);
    cmd.AddValue("duration", "Duration of the emulation in seconds", duration);
    // fixed-hairpin-policy
    cmd.AddValue("group_size",  "Fixed group size", fixed_group_size);
    // hairpin-policy
    cmd.AddValue("coeff",       "QoE Coefficent", qoeCoeff);
    // webrtc policy
    cmd.AddValue("fixed_loss", "Fixed loss rate directly passed to WebRTCPolicy  and WebRTCPolicyStarPolicy", fixed_loss_flag);
    // webrtcstar policy
    cmd.AddValue("order", "The order of the mapping from rtt/t to Deadline-aware Multiplier, supporting 1 or 2", k_order);
    
    cmd.AddValue("param1", "The double_t parameters in baseline", param1);

    cmd.AddValue("cc", "Congestion control algorithm [NOT_USE_CC, GCC]", cc_option);
    cmd.AddValue("vary", "Network varies according to traces", network_variation);
    cmd.AddValue("trace", "Trace file directory", trace);
    cmd.AddValue("interval", "Network condition change interval, in ms", variation_interval);
    cmd.AddValue("window", "Receiver sliding window size, in ms", receiver_wnd);
    cmd.AddValue("settrace", "Receiver feedbacks online traces as network states", set_trace);
    cmd.AddValue("max_fec", "Max FEC rate", max_fec_rate);
    cmd.AddValue("log", "output log directory", logDir);
    cmd.AddValue("isPcapEnabled", "Capture all the packets", isPcapEnabled);
    cmd.Parse (argc, argv);

    std::string dir = logDir + "/" + rtxPolicy + fecPolicy;
    if (fecPolicy == "hairpin" || fecPolicy == "hairpinone") {
      char buf[10];
      snprintf (buf, sizeof (buf), "%.0e", qoeCoeff);
      dir += buf;
    }
    else if (fecPolicy == "fixed" || fecPolicy == "lin" || fecPolicy == "fixedrtx")
      dir += std::to_string (param1).substr (0, 4);

    if (network_variation) {
      std::string traceDir, traceName;
      std::stringstream tracePath (trace);
      std::getline (tracePath, traceDir, '/');
      std::getline (tracePath, traceName, '/');
      dir += "/" + traceName;
    }

    boost::filesystem::create_directories (dir);

    if (max_fec_rate < 0) {
        max_fec_rate = -1;
    }

    if (!boost::filesystem::exists (trace))
      NS_ABORT_MSG ("Trace file does not exist: " + trace);
    
    if (network_variation) {
        std::string tmp;
        std::ifstream trace_file;
        float n = 0.;
        trace_file.open(trace, std::ios::in);
        if (trace_file.fail ()) {
          NS_FATAL_ERROR ("Trace file fail to open!" + trace);
        }
        else {
          while (getline(trace_file, tmp)) {
            n += 1.;
          }
          trace_file.close();
          duration = std::min (duration, (float) (n * variation_interval / 1000.));
          if (duration < 2) {
            NS_FATAL_ERROR ("Trace file too short!" + trace);
          }
        }
    }

    appStop = appStart + duration;    /* emulation duration */

    EnableLogging (LOG_LEVEL_INFO);

    NS_LOG_INFO ("Starting FEC Emulator...");
    NS_LOG_INFO (
        "policy: "          << fecPolicy <<
        ", ddl: "           << delayDdl <<
        "ms, loss: "        << loss_rate * 100 <<
        "%, bw: "           << linkBw <<
        "Mbps, RTT: "       << ms_delay * 2 <<
        "ms, fps: "         << fps <<
        ", bitrate: "       << bitrate <<
        "Mbps, duration: "  << duration <<
        "s, max FEC rate: "  << (max_fec_rate == -1 ? "unset" : (std::to_string(max_fec_rate * 100.0) + "%")) <<
        ", Fixed loss: " << fixed_loss_flag <<
        "\n"
    );
    
    // set FEC policy
    Ptr<FECPolicy> fecPolicyIns;
    if(fecPolicy == "hairpin" || fecPolicy == "hairpinbound") {
        fecPolicyIns = CreateObject<HairpinPolicy> (delayDdl, qoeCoeff, 1, 0);
    } else if(fecPolicy == "hairpinone") {
        fecPolicyIns = CreateObject<HairpinPolicy> (0, qoeCoeff, 1, 0);
    } else if(fecPolicy == "fixed") {
        fecPolicyIns = CreateObject<FixedPolicy> (param1);
    } else if(fecPolicy == "webrtc") {
        fecPolicyIns = CreateObject<WebRTCPolicy> ();
    } else if(fecPolicy == "awebrtc") {
        fecPolicyIns = CreateObject<WebRTCAdaptivePolicy> ();
    // } else if(fecPolicy == "sqrt") {
    //     fecPolicyIns = CreateObject<WebRTCStarPolicy> (0);
    } else if(fecPolicy == "lin") {
        fecPolicyIns = CreateObject<WebRTCStarPolicy> (1, param1);
    // } else if(fecPolicy == "quad") {
    //     fecPolicyIns = CreateObject<WebRTCStarPolicy> (2);
    } else if(fecPolicy == "rtx") {
        fecPolicyIns = CreateObject<RtxOnlyPolicy> ();
    // } else if(fecPolicy == "fec" || fecPolicy == "feconly") {
    //     fecPolicyIns = CreateObject<HairpinPolicy> (delayDdl, (uint8_t) qoeCoeffPow, 0, 0);
    } else if(fecPolicy == "bolot") {
        fecPolicyIns = CreateObject<BolotPolicy> ();
    } else if(fecPolicy == "usf") {
        fecPolicyIns = CreateObject<UsfPolicy> ();
    } else if(fecPolicy == "fixedrtx") {
        fecPolicyIns = CreateObject<FixedRtxPolicy> (param1);
    } else if(fecPolicy == "tokenrtx") {
        fecPolicyIns = CreateObject<TokenRtxPolicy> ();
    } else {
        NS_ASSERT_MSG(false, "FECPolicy must be one of [hairpin, hairpinbound, fixed, webrtc, awebrtc, rtx, fec].");
    }
    fecPolicyIns->SetMaxFECRate (max_fec_rate);
    if (fixed_loss_flag) {
      fecPolicyIns->SetFixedLoss (loss_rate);
    }


  Config::SetDefault ("ns3::RateErrorModel::ErrorRate", DoubleValue (loss_rate));
	Config::SetDefault ("ns3::RateErrorModel::ErrorUnit", StringValue ("ERROR_UNIT_PACKET"));


  NodeContainer nodes = BuildExampleTopo (linkBw * 1e6, ms_delay, msQDelay, dir, isPcapEnabled);
  NS_LOG_INFO("Topology successfully built...");

  // DEBUG("node0: " << nodes.Get(0)->GetNDevices() <<  ", node1: " << nodes.Get(1)->GetNDevices() <<  ", node2: " << nodes.Get(2)->GetNDevices());

  NS_LOG_INFO ("Installing application...");
  NS_LOG_INFO ("track network dynamics: " << network_variation);

  switch (cc_option)
  {
  case NOT_USE_CC:
    if (network_variation) {
      /* Set network propagation delay and error rate */
      /* according to frame-level traces */
      BandwidthTrace (nodes.Get(0), nodes.Get(1), trace, false, variation_interval, true);
    }
    NS_LOG_INFO("CC algorithm: NOT_USE_CC");
    break;

  case GCC:
  case NADA:
    if (network_variation) {
      BandwidthTrace (nodes.Get(0), nodes.Get(1), trace, true, variation_interval, true);
    }
    NS_LOG_INFO("CC algorithm: " << cc_option);
    Config::SetDefault ("ns3::NetworkPacket::MaxPacketSize", UintegerValue (260));
    break;

  default:
    NS_ASSERT_MSG(false, "Invalid CC algorithm selected.");
    break;
  }

  InstallApp (
    nodes.Get(0), nodes.Get(2),
    port, appStart, appStop, fecPolicyIns, rtxPolicy,
    fps, variation_interval, delayDdl, bitrate * 1e6, cc_option,
    ms_delay * 2, bitrate, loss_rate, receiver_wnd, set_trace, trace, dir);

  Simulator::Run ();
  Simulator::Stop (Seconds (appStop + 2));
  Simulator::Destroy ();
}
// 