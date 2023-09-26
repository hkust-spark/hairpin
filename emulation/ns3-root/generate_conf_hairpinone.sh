fname="wifi-hairpinone.conf"
rm ${fname}
for trace in `ls traces | grep wifi`
do
    conf="fec-emulate --vary=1 --trace=traces/${trace}"
    rtx=dupack
    for coeff in 1 2 4 7
    do
        echo '"'${conf}' --fecPolicy=hairpinone --rtxPolicy='${rtx}' --coeff='${coeff}'",logs/'${rtx}'hairpinone'${coeff}'/'${trace}'/output.tr' >> ${fname}
    done
done
