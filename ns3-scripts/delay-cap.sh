awk -F '[ m]' '{
    if ($2 < 80 && $2 > 2 && $4 < 1) {
        print $0
    }
}' $2/$1 > $1
