fname="wifi-ddl.conf"
rm ${fname}
for trace in `ls traces | grep wifi`
do
    for ddl in 50 200
    do
    conf="rtc-test --vary=1 --log=logs-ddl${ddl} --ddl=${ddl} --trace=traces/${trace}"
        for rtx in dupack pto
        do
            echo '"'${conf}' --fecPolicy=rtx --rtxPolicy='${rtx}'",logs-ddl'${ddl}'/'${rtx}'rtx/'${trace}'/output.log' >> ${fname}
            echo '"'${conf}' --fecPolicy=bolot --rtxPolicy='${rtx}'",logs-ddl'${ddl}'/'${rtx}'bolot/'${trace}'/output.log' >> ${fname}
            echo '"'${conf}' --fecPolicy=usf --rtxPolicy='${rtx}'",logs-ddl'${ddl}'/'${rtx}'usf/'${trace}'/output.log' >> ${fname}
            echo '"'${conf}' --fecPolicy=webrtc --rtxPolicy='${rtx}'",logs-ddl'${ddl}'/'${rtx}'webrtc/'${trace}'/output.log' >> ${fname}
            echo '"'${conf}' --fecPolicy=awebrtc --rtxPolicy='${rtx}'",logs-ddl'${ddl}'/'${rtx}'awebrtc/'${trace}'/output.log' >> ${fname}   
        done
        
        rtx=dupack
        coeff=1e-7
        echo '"'${conf}' --fecPolicy=hairpin --rtxPolicy='${rtx}' --coeff='${coeff}'",logs-ddl'${ddl}'/'${rtx}'hairpin'${coeff}'/'${trace}'/output.log' >> ${fname}
    done
done
