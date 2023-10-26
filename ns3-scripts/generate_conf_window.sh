fname="wifi-window.conf"
rm ${fname}
for trace in `ls traces | grep wifi`
do
    for window in 17 34 68 136
    do
        logdir="logs-window"${window}
        conf="fec-emulate --vary=1 --log=${logdir} --window=${window} --trace=traces/${trace}"
        rtx=dupack
        coeff=1e-7
        echo '"'${conf}' --fecPolicy=hairpin --rtxPolicy='${rtx}' --coeff='${coeff}'",'${logdir}'/'${rtx}'hairpin'${coeff}'/'${trace}'/output.tr' >> ${fname}
    done
done
