import pstats
sts = pstats.Stats('main.prof')
sts.strip_dirs().sort_stats(-1).print_stats()