fname="test-wifidupack.conf"
rm ${fname}
for trace in `ls traces-wifi`
do
    conf="rtc-test --vary=1 --trace=traces-wifi/${trace}"
    for rtx in dupack
    do
        echo '"'${conf}' --fecPolicy=rtx --rtxPolicy='${rtx}'",logs/'${rtx}'rtx/'${trace}'/output.log' >> ${fname}
        echo '"'${conf}' --fecPolicy=bolot --rtxPolicy='${rtx}'",logs/'${rtx}'bolot/'${trace}'/output.log' >> ${fname}
        echo '"'${conf}' --fecPolicy=usf --rtxPolicy='${rtx}'",logs/'${rtx}'usf/'${trace}'/output.log' >> ${fname}
        echo '"'${conf}' --fecPolicy=webrtc --rtxPolicy='${rtx}'",logs/'${rtx}'webrtc/'${trace}'/output.log' >> ${fname}
        echo '"'${conf}' --fecPolicy=awebrtc --rtxPolicy='${rtx}'",logs/'${rtx}'awebrtc/'${trace}'/output.log' >> ${fname}
    done
    
    rtx=dupack
    for coeff in 1e-7
    do
        echo '"'${conf}' --fecPolicy=hairpin --rtxPolicy='${rtx}' --coeff='${coeff}'",logs/'${rtx}'hairpin'${coeff}'/'${trace}'/output.log' >> ${fname}
    done
done
