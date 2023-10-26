fname="wifi-cc.conf"
rm ${fname}
for trace in `ls traces | grep wifi`
do
    for cc in 1 2
    do
        logdir="logs-cc"${cc}
        conf="fec-emulate --vary=1 --log=${logdir} --cc=${cc} --trace=traces/${trace}"
        for rtx in dupack pto
        do
            echo '"'${conf}' --fecPolicy=rtx --rtxPolicy='${rtx}'",'${logdir}'/'${rtx}'rtx/'${trace}'/output.tr' >> ${fname}
            echo '"'${conf}' --fecPolicy=bolot --rtxPolicy='${rtx}'",'${logdir}'/'${rtx}'bolot/'${trace}'/output.tr' >> ${fname}
            echo '"'${conf}' --fecPolicy=usf --rtxPolicy='${rtx}'",'${logdir}'/'${rtx}'usf/'${trace}'/output.tr' >> ${fname}
            echo '"'${conf}' --fecPolicy=webrtc --rtxPolicy='${rtx}'",'${logdir}'/'${rtx}'webrtc/'${trace}'/output.tr' >> ${fname}
            echo '"'${conf}' --fecPolicy=awebrtc --rtxPolicy='${rtx}'",'${logdir}'/'${rtx}'awebrtc/'${trace}'/output.tr' >> ${fname}   
        done
        
        rtx=dupack
        coeff=1e-7
        echo '"'${conf}' --fecPolicy=hairpin --rtxPolicy='${rtx}' --coeff='${coeff}'",'${logdir}'/'${rtx}'hairpin'${coeff}'/'${trace}'/output.tr' >> ${fname}
    done
done
