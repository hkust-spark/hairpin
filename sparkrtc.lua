local sparkrtc_protocol = Proto("SparkRTC", "SparkRTC Protocol")
local proto_name = "SparkRTC"

-- ---- Fields ---- --
-- NetworkPacket
local packet_type = ProtoField.uint32(proto_name .. ".packet_type", "packet_type", base.DEC)
-- VideoPacket
local encode_type = ProtoField.uint64(proto_name .. ".encode_time", "encode_time", base.DEC)
local global_id = ProtoField.uint16(proto_name .. ".global_id", "global_id", base.DEC)
local group_id = ProtoField.uint32(proto_name .. ".group_id", "group_id", base.DEC)
local group_data_num = ProtoField.uint16(proto_name .. ".group_data_num", "group_data_num", base.DEC)
local group_fec_num = ProtoField.uint16(proto_name .. ".group_fec_num", "group_fec_num", base.DEC)
local pkt_id_in_group = ProtoField.uint16(proto_name .. ".pkt_id_in_group", "pkt_id_in_group", base.DEC)
local batch_id = ProtoField.uint32(proto_name .. ".batch_id", "batch_id", base.DEC)
local batch_data_num = ProtoField.uint16(proto_name .. ".batch_data_num", "batch_data_num", base.DEC)
local batch_fec_num = ProtoField.uint16(proto_name .. ".batch_fec_num", "batch_fec_num", base.DEC)
local pkt_id_in_batch = ProtoField.uint16(proto_name .. ".pkt_id_in_batch", "pkt_id_in_batch", base.DEC)
local tx_count = ProtoField.uint8(proto_name .. ".tx_count", "tx_count", base.DEC)
-- DataPacket / DubFECPacket
local frame_id = ProtoField.uint32(proto_name .. ".frame_id", "frame_id", base.DEC)
local frame_pkt_num = ProtoField.uint16(proto_name .. ".frame_pkt_num", "frame_pkt_num", base.DEC)
local pkt_id_in_frame = ProtoField.uint16(proto_name .. ".pkt_id_in_frame", "pkt_id_in_frame", base.DEC)
-- FECPacket
local fec_data_count = ProtoField.uint16(proto_name .. ".fec_data_count", "fec_data_count", base.DEC)
-- RTXReqPacket
local rtx_req_count = ProtoField.uint16(proto_name .. ".rtx_req_count", "rtx_req_count", base.DEC)
-- NetStatePacket
local loss_rate = ProtoField.uint16(proto_name .. ".loss_rate", "loss_rate", base.DEC)
local throughput = ProtoField.uint16(proto_name .. ".throughput", "throughput", base.DEC)
local group_delay = ProtoField.uint16(proto_name .. ".group_delay", "group_delay", base.DEC)
local loss_seq_size = ProtoField.uint16(proto_name .. ".loss_seq_size", "loss_seq_size", base.DEC)



sparkrtc_protocol.fields = {
    packet_type,
    encode_type, global_id,
    group_id, group_data_num, group_fec_num, pkt_id_in_group,
    batch_id, batch_data_num, batch_fec_num, pkt_id_in_batch, tx_count,
    frame_id, frame_pkt_num, pkt_id_in_frame,
    fec_data_count,
    rtx_req_count,
    loss_rate, throughput, group_delay, loss_seq_size
}

function get_packet_type(opcode)
    local opcode_name = "Unknown"
        if opcode == 0 then opcode_name = "DATA_PKT"
    elseif opcode == 1 then opcode_name = "DUP_FEC_PKT"
    elseif opcode == 2 then opcode_name = "FEC_PKT"
    elseif opcode == 3 then opcode_name = "RTX_REQ_PKT"
    elseif opcode == 4 then opcode_name = "NETSTATE_PKT" end
    return opcode_name
end

function sparkrtc_protocol.dissector(buffer, pinfo, tree)
    length = buffer:len()
    if length == 0 then return end

    pinfo.cols.protocol = sparkrtc_protocol.name

    local subtree = tree:add(sparkrtc_protocol, buffer(), "SparkRTC Protocol Data")

    -- packet_type
    local packet_type_code = buffer(0, 4):uint()
    local packet_type_name = get_packet_type(packet_type_code)
    subtree:add(packet_type, buffer(0, 4)):append_text(" (" .. packet_type_name .. ")")

    -- Payload
    if packet_type_name == "DATA_PKT" or packet_type_name == "DUP_FEC_PKT" or packet_type_name == "FEC_PKT" then
        -- VideoPacket
        local subtree_video = subtree:add(sparkrtc_protocol, buffer(), "Video")
        subtree_video:add(encode_type, buffer(4, 8))
        subtree_video:add(global_id, buffer(12, 2))
        subtree_video:add(group_id, buffer(14, 4))
        subtree_video:add(group_data_num, buffer(18, 2))
        subtree_video:add(group_fec_num, buffer(20, 2))
        subtree_video:add(pkt_id_in_group, buffer(22, 2))
        subtree_video:add(batch_id, buffer(24, 4))
        subtree_video:add(batch_data_num, buffer(28, 2))
        subtree_video:add(batch_fec_num, buffer(30, 2))
        subtree_video:add(pkt_id_in_batch, buffer(32, 2))
        subtree_video:add(tx_count, buffer(34, 1))
        if packet_type_name == "DATA_PKT" or packet_type_name == "DUP_FEC_PKT" then
            local subtree_data = subtree_video:add(sparkrtc_protocol, buffer(), "Frame")
            -- DataPacket / DupFECPacket
            subtree_data:add(frame_id, buffer(35, 4))
            subtree_data:add(frame_pkt_num, buffer(39, 2))
            subtree_data:add(pkt_id_in_frame, buffer(41, 2))
        elseif packet_type_name == "FEC_PKT" then
            local subtree_fec = subtree_video:add(sparkrtc_protocol, buffer(), "FEC")
            -- DataPacket / DupFECPacket
            subtree_fec:add(fec_data_count, buffer(35, 2))
        end
    elseif packet_type_name == "RTX_REQ_PKT" then
        -- RtxRequestPacket
        local subtree_rtx_req = subtree:add(sparkrtc_protocol, buffer(), "RTXReq")
        subtree_rtx_req:add(rtx_req_count, buffer(4, 2))
    elseif packet_type_name == "NETSTATE_PKT" then
        -- NetStatePacket
        local subtree_state = subtree:add(sparkrtc_protocol, buffer(), "NetState")
        subtree_state:add(loss_rate, buffer(4, 2))
        subtree_state:add(throughput, buffer(6, 2))
        subtree_state:add(group_delay, buffer(8, 2))
        subtree_state:add(loss_seq_size, buffer(10, 2))
    end
end

local udp_port = DissectorTable.get("udp.port")
udp_port:add(8000, sparkrtc_protocol)