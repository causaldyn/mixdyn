# The nested model's long-term share on long-term null worlds

Recorded on 2026-11-07 in two runs from the repository's root, in its locked environment:

```
JAX_PLATFORMS=cpu python -m validation.long_term_null --worlds 20
JAX_PLATFORMS=cpu python -m validation.long_term_null --start 20
JAX_PLATFORMS=cpu python -m validation.long_term_null --tally validation/long_term_null.md
```

The first read worlds 0 to 19 under the three readings; the second extended the nested model's on
the worlds whose brand measure sales do not read to 100, worlds 20 to 99, on the same seeds and
settings; the third printed the summary lines after the table from its rows. The table is the
first two runs' rows. The worlds, the models, the seeds and what "covers 0" reads are
`validation/long_term_null.py`'s docstring; the decision it informs is ADR 0014's amendment of
2026-11-06.

The criterion: the nested model's share covers 0 on at least 90 % of the worlds whose brand
measure sales do not read, met where the rate's 95 % Clopper-Pearson interval holds 0.9 or lies
above it. The first 20 worlds gave 17, and the extension was run to tell 0.85 from 0.9: 91 of 100,
[0.836, 0.958], which holds 0.9, and does not rule out a rate down to 0.836 either.
`tests/test_long_term_null.py` reads the first world of each kind and checks this record against
the criterion. The second table, below the first, reads the nested model with stage one's return
prior half-normal (ADR 0076) against the criterion set for it, read the same way: its share covers
0 on at least 90 % of the worlds whose brand measure television does not build.

`SE` in the last lines is the rate's Monte Carlo standard error over the worlds, `sqrt(p (1 - p) /
n)`, which is 0 at a rate of 0 or 1; the Clopper-Pearson interval is the one to read there.

`low`, `median` and `high` are the 5 %, 50 % and 95 % quantiles of the share over the model's 400
draws; `divergent` counts the fit's divergent transitions, both stages' for the nested model. A
world's number is its place in the seeds, and the two kinds of world of one number draw the same
spend, price and noise.

| model | cut | world | low | median | high | covers 0 | divergent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nested | unread | 0 | 0.0053 | 0.0303 | 0.0891 | yes | 0 |
| dual adstock | unread | 0 | 0.0745 | 0.2053 | 0.3997 | no | 0 |
| nested | unbuilt | 0 | 0.0402 | 0.1118 | 0.2325 | no | 0 |
| nested | unread | 1 | 0.0036 | 0.0523 | 0.1945 | yes | 0 |
| dual adstock | unread | 1 | 0.0631 | 0.1916 | 0.4165 | no | 0 |
| nested | unbuilt | 1 | 0.0618 | 0.2290 | 0.4409 | no | 0 |
| nested | unread | 2 | 0.0159 | 0.0661 | 0.1652 | yes | 0 |
| dual adstock | unread | 2 | 0.1120 | 0.3043 | 0.5637 | no | 0 |
| nested | unbuilt | 2 | 0.1020 | 0.1896 | 0.3380 | no | 0 |
| nested | unread | 3 | 0.0049 | 0.0433 | 0.1492 | yes | 0 |
| dual adstock | unread | 3 | 0.0998 | 0.2687 | 0.4729 | no | 0 |
| nested | unbuilt | 3 | 0.0640 | 0.1834 | 0.3975 | no | 0 |
| nested | unread | 4 | 0.0021 | 0.0247 | 0.1114 | yes | 0 |
| dual adstock | unread | 4 | 0.1078 | 0.2878 | 0.5390 | no | 0 |
| nested | unbuilt | 4 | 0.0723 | 0.1748 | 0.4258 | no | 0 |
| nested | unread | 5 | 0.0013 | 0.0208 | 0.1126 | yes | 0 |
| dual adstock | unread | 5 | 0.0670 | 0.1919 | 0.4024 | no | 0 |
| nested | unbuilt | 5 | 0.0659 | 0.1462 | 0.3229 | no | 0 |
| nested | unread | 6 | 0.0391 | 0.1826 | 0.3964 | no | 0 |
| dual adstock | unread | 6 | 0.0904 | 0.2231 | 0.4517 | no | 0 |
| nested | unbuilt | 6 | 0.1104 | 0.2640 | 0.5156 | no | 0 |
| nested | unread | 7 | 0.0166 | 0.0828 | 0.2470 | yes | 0 |
| dual adstock | unread | 7 | 0.0978 | 0.2402 | 0.4400 | no | 0 |
| nested | unbuilt | 7 | 0.0749 | 0.1888 | 0.4336 | no | 0 |
| nested | unread | 8 | 0.0221 | 0.1118 | 0.2667 | no | 0 |
| dual adstock | unread | 8 | 0.1089 | 0.2719 | 0.4924 | no | 0 |
| nested | unbuilt | 8 | 0.0984 | 0.2310 | 0.4581 | no | 0 |
| nested | unread | 9 | 0.0006 | 0.0518 | 0.1981 | yes | 0 |
| dual adstock | unread | 9 | 0.0850 | 0.1997 | 0.4002 | no | 0 |
| nested | unbuilt | 9 | 0.0580 | 0.1670 | 0.4191 | no | 0 |
| nested | unread | 10 | 0.0027 | 0.0270 | 0.0971 | yes | 0 |
| dual adstock | unread | 10 | 0.0963 | 0.2520 | 0.4194 | no | 0 |
| nested | unread | 11 | 0.0278 | 0.0945 | 0.1872 | no | 0 |
| dual adstock | unread | 11 | 0.0645 | 0.1551 | 0.2628 | no | 0 |
| nested | unread | 12 | 0.0015 | 0.0137 | 0.0711 | yes | 0 |
| dual adstock | unread | 12 | 0.1231 | 0.2650 | 0.4579 | no | 0 |
| nested | unread | 13 | 0.0059 | 0.0491 | 0.1708 | yes | 0 |
| dual adstock | unread | 13 | 0.0882 | 0.2054 | 0.3927 | no | 0 |
| nested | unread | 14 | 0.0044 | 0.0404 | 0.1453 | yes | 0 |
| dual adstock | unread | 14 | 0.0645 | 0.1831 | 0.3609 | no | 0 |
| nested | unread | 15 | 0.0092 | 0.0764 | 0.2306 | yes | 0 |
| dual adstock | unread | 15 | 0.1250 | 0.2612 | 0.4204 | no | 0 |
| nested | unread | 16 | 0.0045 | 0.0417 | 0.1426 | yes | 0 |
| dual adstock | unread | 16 | 0.1225 | 0.3325 | 0.5686 | no | 0 |
| nested | unread | 17 | 0.0011 | 0.0136 | 0.0532 | yes | 0 |
| dual adstock | unread | 17 | 0.0663 | 0.1711 | 0.3128 | no | 0 |
| nested | unread | 18 | 0.0187 | 0.0919 | 0.3020 | yes | 0 |
| dual adstock | unread | 18 | 0.1400 | 0.3571 | 0.6094 | no | 0 |
| nested | unread | 19 | 0.0119 | 0.1183 | 0.2929 | yes | 0 |
| dual adstock | unread | 19 | 0.0944 | 0.2351 | 0.4523 | no | 0 |
| nested | unread | 20 | 0.0047 | 0.0554 | 0.1657 | yes | 0 |
| nested | unread | 21 | 0.0037 | 0.0487 | 0.1826 | yes | 0 |
| nested | unread | 22 | 0.0026 | 0.0307 | 0.1158 | yes | 0 |
| nested | unread | 23 | 0.0016 | 0.0226 | 0.1012 | yes | 0 |
| nested | unread | 24 | 0.0011 | 0.0195 | 0.0781 | yes | 0 |
| nested | unread | 25 | 0.0021 | 0.0262 | 0.0957 | yes | 0 |
| nested | unread | 26 | 0.0107 | 0.0682 | 0.2131 | yes | 0 |
| nested | unread | 27 | 0.0116 | 0.0930 | 0.2094 | yes | 0 |
| nested | unread | 28 | 0.0145 | 0.1076 | 0.2902 | yes | 0 |
| nested | unread | 29 | 0.0026 | 0.0297 | 0.1078 | yes | 0 |
| nested | unread | 30 | 0.0012 | 0.0426 | 0.1429 | yes | 0 |
| nested | unread | 31 | 0.0021 | 0.0305 | 0.1196 | yes | 0 |
| nested | unread | 32 | 0.0103 | 0.0872 | 0.2181 | yes | 0 |
| nested | unread | 33 | 0.0036 | 0.0365 | 0.1556 | yes | 0 |
| nested | unread | 34 | 0.0406 | 0.1151 | 0.2629 | no | 0 |
| nested | unread | 35 | 0.0017 | 0.0362 | 0.1297 | yes | 0 |
| nested | unread | 36 | 0.0049 | 0.0626 | 0.1847 | yes | 0 |
| nested | unread | 37 | 0.0086 | 0.0733 | 0.2218 | yes | 0 |
| nested | unread | 38 | 0.0030 | 0.0345 | 0.1160 | yes | 0 |
| nested | unread | 39 | 0.0023 | 0.0260 | 0.1315 | yes | 0 |
| nested | unread | 40 | 0.0037 | 0.0442 | 0.1594 | yes | 0 |
| nested | unread | 41 | 0.0213 | 0.1289 | 0.2781 | no | 0 |
| nested | unread | 42 | 0.0325 | 0.1558 | 0.3095 | no | 0 |
| nested | unread | 43 | 0.0022 | 0.0214 | 0.0938 | yes | 0 |
| nested | unread | 44 | 0.0021 | 0.0238 | 0.0989 | yes | 0 |
| nested | unread | 45 | 0.0049 | 0.0533 | 0.1983 | yes | 0 |
| nested | unread | 46 | 0.0009 | 0.0210 | 0.0915 | yes | 0 |
| nested | unread | 47 | 0.0026 | 0.0269 | 0.1287 | yes | 0 |
| nested | unread | 48 | 0.0063 | 0.0440 | 0.1559 | yes | 0 |
| nested | unread | 49 | 0.0016 | 0.0255 | 0.1071 | yes | 0 |
| nested | unread | 50 | 0.0215 | 0.0970 | 0.2547 | no | 0 |
| nested | unread | 51 | 0.0024 | 0.0296 | 0.1434 | yes | 0 |
| nested | unread | 52 | 0.0179 | 0.0869 | 0.2349 | yes | 0 |
| nested | unread | 53 | 0.0238 | 0.1797 | 0.4152 | no | 0 |
| nested | unread | 54 | 0.0124 | 0.0705 | 0.2040 | yes | 0 |
| nested | unread | 55 | 0.0121 | 0.0773 | 0.2221 | yes | 0 |
| nested | unread | 56 | 0.0022 | 0.0220 | 0.1021 | yes | 0 |
| nested | unread | 57 | 0.0018 | 0.0257 | 0.1041 | yes | 0 |
| nested | unread | 58 | 0.0021 | 0.0306 | 0.1128 | yes | 0 |
| nested | unread | 59 | 0.0067 | 0.0514 | 0.1892 | yes | 0 |
| nested | unread | 60 | 0.0013 | 0.0245 | 0.1222 | yes | 0 |
| nested | unread | 61 | 0.0024 | 0.0255 | 0.1084 | yes | 0 |
| nested | unread | 62 | 0.0053 | 0.0528 | 0.1697 | yes | 0 |
| nested | unread | 63 | 0.0012 | 0.0312 | 0.1229 | yes | 0 |
| nested | unread | 64 | 0.0071 | 0.0663 | 0.1716 | yes | 0 |
| nested | unread | 65 | 0.0012 | 0.0200 | 0.1168 | yes | 0 |
| nested | unread | 66 | 0.0017 | 0.0322 | 0.1243 | yes | 0 |
| nested | unread | 67 | 0.0006 | 0.0077 | 0.0354 | yes | 0 |
| nested | unread | 68 | 0.0086 | 0.0521 | 0.1517 | yes | 0 |
| nested | unread | 69 | 0.0018 | 0.0233 | 0.1327 | yes | 0 |
| nested | unread | 70 | 0.0039 | 0.0482 | 0.1657 | yes | 0 |
| nested | unread | 71 | 0.0026 | 0.0293 | 0.1304 | yes | 0 |
| nested | unread | 72 | 0.0052 | 0.0642 | 0.2104 | yes | 0 |
| nested | unread | 73 | 0.0012 | 0.0115 | 0.0584 | yes | 0 |
| nested | unread | 74 | 0.0070 | 0.0491 | 0.1662 | yes | 0 |
| nested | unread | 75 | 0.0068 | 0.0737 | 0.1959 | yes | 0 |
| nested | unread | 76 | 0.0155 | 0.0955 | 0.2428 | yes | 0 |
| nested | unread | 77 | 0.0219 | 0.1323 | 0.3013 | no | 0 |
| nested | unread | 78 | 0.0025 | 0.0283 | 0.1130 | yes | 0 |
| nested | unread | 79 | 0.0067 | 0.0499 | 0.1329 | yes | 0 |
| nested | unread | 80 | 0.0022 | 0.0235 | 0.0934 | yes | 0 |
| nested | unread | 81 | 0.0101 | 0.0747 | 0.2489 | yes | 0 |
| nested | unread | 82 | 0.0041 | 0.0451 | 0.1490 | yes | 0 |
| nested | unread | 83 | 0.0114 | 0.0893 | 0.3195 | yes | 0 |
| nested | unread | 84 | 0.0040 | 0.0359 | 0.1313 | yes | 0 |
| nested | unread | 85 | 0.0044 | 0.0593 | 0.1854 | yes | 0 |
| nested | unread | 86 | 0.0026 | 0.0341 | 0.1351 | yes | 0 |
| nested | unread | 87 | 0.0017 | 0.0215 | 0.1128 | yes | 0 |
| nested | unread | 88 | 0.0045 | 0.0387 | 0.1380 | yes | 0 |
| nested | unread | 89 | 0.0089 | 0.0745 | 0.1872 | yes | 0 |
| nested | unread | 90 | 0.0019 | 0.0228 | 0.1166 | yes | 0 |
| nested | unread | 91 | 0.0008 | 0.0120 | 0.0590 | yes | 0 |
| nested | unread | 92 | 0.0024 | 0.0200 | 0.0827 | yes | 0 |
| nested | unread | 93 | 0.0094 | 0.0926 | 0.2463 | yes | 0 |
| nested | unread | 94 | 0.0052 | 0.0531 | 0.1822 | yes | 0 |
| nested | unread | 95 | 0.0015 | 0.0241 | 0.0924 | yes | 0 |
| nested | unread | 96 | 0.0104 | 0.0824 | 0.1991 | yes | 0 |
| nested | unread | 97 | 0.0021 | 0.0316 | 0.1075 | yes | 0 |
| nested | unread | 98 | 0.0047 | 0.0354 | 0.1385 | yes | 0 |
| nested | unread | 99 | 0.0069 | 0.0476 | 0.1617 | yes | 0 |

- nested, unread: covers 0 in 91 of 100 worlds, 0.91 (SE 0.029; 95 % Clopper-Pearson [0.836, 0.958])
- dual adstock, unread: covers 0 in 0 of 20 worlds, 0.00 (SE 0.000; 95 % Clopper-Pearson [0.000, 0.168])
- nested, unbuilt: covers 0 in 0 of 10 worlds, 0.00 (SE 0.000; 95 % Clopper-Pearson [0.000, 0.308])

## The nested model with a half-normal stage one

Recorded on 2026-11-07 in two runs from the repository's root, then tallied as the first table:

```
JAX_PLATFORMS=cpu python -m validation.long_term_null --models "nested half-normal" --worlds 20 --unbuilt 20
JAX_PLATFORMS=cpu python -m validation.long_term_null --models "nested half-normal" --start 20 --worlds 20 --unbuilt 100
JAX_PLATFORMS=cpu python -m validation.long_term_null --tally validation/long_term_null.md
```

The model is the first table's nested model with television's return on the searches
`HalfNormalRoi()` at the searches' scale, its default scale times their mean over sales', on the
same seeds and settings. The first run read worlds 0 to 19 of both kinds; the second extended the
worlds whose brand television does not build to 100, worlds 20 to 99, and was started at world 19,
whose two rows it read again to the digit. Where television does not build the brand, the share
covers 0 on 87 of 100 worlds, [0.788, 0.929], which holds 0.9: the criterion is met as the first
table's is, and the interval does not rule out a rate down to 0.788 either. The first 20 worlds
gave 15, and the extension was run to tell 0.75 from 0.9. Where sales do not read the brand, it
covers 0 on 18 of 20, against the first table's 17 on the same 20.

| model | cut | world | low | median | high | covers 0 | divergent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nested half-normal | unread | 0 | 0.0050 | 0.0278 | 0.0837 | yes | 0 |
| nested half-normal | unbuilt | 0 | 0.0060 | 0.0429 | 0.1312 | yes | 0 |
| nested half-normal | unread | 1 | 0.0035 | 0.0421 | 0.1530 | yes | 0 |
| nested half-normal | unbuilt | 1 | 0.0048 | 0.0704 | 0.3080 | yes | 0 |
| nested half-normal | unread | 2 | 0.0137 | 0.0650 | 0.1488 | yes | 0 |
| nested half-normal | unbuilt | 2 | 0.0430 | 0.1519 | 0.2876 | no | 0 |
| nested half-normal | unread | 3 | 0.0042 | 0.0416 | 0.1312 | yes | 0 |
| nested half-normal | unbuilt | 3 | 0.0058 | 0.0693 | 0.2802 | yes | 0 |
| nested half-normal | unread | 4 | 0.0022 | 0.0211 | 0.0857 | yes | 0 |
| nested half-normal | unbuilt | 4 | 0.0132 | 0.0789 | 0.2362 | yes | 0 |
| nested half-normal | unread | 5 | 0.0009 | 0.0169 | 0.0782 | yes | 0 |
| nested half-normal | unbuilt | 5 | 0.0068 | 0.0690 | 0.2169 | yes | 0 |
| nested half-normal | unread | 6 | 0.0356 | 0.1472 | 0.3075 | no | 0 |
| nested half-normal | unbuilt | 6 | 0.0211 | 0.1216 | 0.3642 | no | 0 |
| nested half-normal | unread | 7 | 0.0149 | 0.0682 | 0.1888 | yes | 0 |
| nested half-normal | unbuilt | 7 | 0.0092 | 0.0870 | 0.2712 | yes | 0 |
| nested half-normal | unread | 8 | 0.0196 | 0.0921 | 0.2478 | yes | 0 |
| nested half-normal | unbuilt | 8 | 0.0217 | 0.1408 | 0.3276 | no | 0 |
| nested half-normal | unread | 9 | 0.0005 | 0.0401 | 0.1556 | yes | 0 |
| nested half-normal | unbuilt | 9 | 0.0021 | 0.0314 | 0.1872 | yes | 0 |
| nested half-normal | unread | 10 | 0.0025 | 0.0239 | 0.1038 | yes | 0 |
| nested half-normal | unbuilt | 10 | 0.0074 | 0.0708 | 0.2135 | yes | 0 |
| nested half-normal | unread | 11 | 0.0275 | 0.0894 | 0.1782 | no | 0 |
| nested half-normal | unbuilt | 11 | 0.0080 | 0.0502 | 0.1606 | yes | 0 |
| nested half-normal | unread | 12 | 0.0011 | 0.0115 | 0.0569 | yes | 0 |
| nested half-normal | unbuilt | 12 | 0.0207 | 0.1258 | 0.3062 | no | 0 |
| nested half-normal | unread | 13 | 0.0049 | 0.0438 | 0.1462 | yes | 0 |
| nested half-normal | unbuilt | 13 | 0.0085 | 0.0731 | 0.2757 | yes | 0 |
| nested half-normal | unread | 14 | 0.0032 | 0.0360 | 0.1232 | yes | 0 |
| nested half-normal | unbuilt | 14 | 0.0029 | 0.0478 | 0.2039 | yes | 0 |
| nested half-normal | unread | 15 | 0.0060 | 0.0602 | 0.1774 | yes | 0 |
| nested half-normal | unbuilt | 15 | 0.0033 | 0.0546 | 0.2438 | yes | 0 |
| nested half-normal | unread | 16 | 0.0030 | 0.0297 | 0.1148 | yes | 0 |
| nested half-normal | unbuilt | 16 | 0.0056 | 0.0696 | 0.3413 | yes | 0 |
| nested half-normal | unread | 17 | 0.0009 | 0.0132 | 0.0525 | yes | 0 |
| nested half-normal | unbuilt | 17 | 0.0031 | 0.0359 | 0.1424 | yes | 0 |
| nested half-normal | unread | 18 | 0.0128 | 0.0768 | 0.2029 | yes | 0 |
| nested half-normal | unbuilt | 18 | 0.0064 | 0.1254 | 0.4141 | yes | 0 |
| nested half-normal | unread | 19 | 0.0077 | 0.0880 | 0.2183 | yes | 0 |
| nested half-normal | unbuilt | 19 | 0.0921 | 0.2952 | 0.5234 | no | 0 |
| nested half-normal | unbuilt | 20 | 0.0121 | 0.0798 | 0.2696 | yes | 0 |
| nested half-normal | unbuilt | 21 | 0.0150 | 0.1052 | 0.3225 | yes | 0 |
| nested half-normal | unbuilt | 22 | 0.0058 | 0.0629 | 0.2214 | yes | 0 |
| nested half-normal | unbuilt | 23 | 0.0067 | 0.0486 | 0.1485 | yes | 0 |
| nested half-normal | unbuilt | 24 | 0.0064 | 0.0474 | 0.1438 | yes | 0 |
| nested half-normal | unbuilt | 25 | 0.0065 | 0.0662 | 0.2372 | yes | 0 |
| nested half-normal | unbuilt | 26 | 0.0033 | 0.0386 | 0.1839 | yes | 0 |
| nested half-normal | unbuilt | 27 | 0.0054 | 0.0684 | 0.2556 | yes | 0 |
| nested half-normal | unbuilt | 28 | 0.0043 | 0.0607 | 0.2573 | yes | 0 |
| nested half-normal | unbuilt | 29 | 0.0051 | 0.0630 | 0.2290 | yes | 0 |
| nested half-normal | unbuilt | 30 | 0.0488 | 0.1863 | 0.3571 | no | 0 |
| nested half-normal | unbuilt | 31 | 0.0045 | 0.0460 | 0.1974 | yes | 0 |
| nested half-normal | unbuilt | 32 | 0.0021 | 0.0373 | 0.1642 | yes | 0 |
| nested half-normal | unbuilt | 33 | 0.0167 | 0.1458 | 0.3473 | yes | 0 |
| nested half-normal | unbuilt | 34 | 0.0269 | 0.1259 | 0.2845 | no | 0 |
| nested half-normal | unbuilt | 35 | 0.0053 | 0.0685 | 0.1898 | yes | 0 |
| nested half-normal | unbuilt | 36 | 0.0076 | 0.0852 | 0.2898 | yes | 0 |
| nested half-normal | unbuilt | 37 | 0.0180 | 0.1326 | 0.3342 | yes | 0 |
| nested half-normal | unbuilt | 38 | 0.0031 | 0.0519 | 0.2112 | yes | 0 |
| nested half-normal | unbuilt | 39 | 0.0039 | 0.0693 | 0.2436 | yes | 0 |
| nested half-normal | unbuilt | 40 | 0.0069 | 0.0798 | 0.3023 | yes | 0 |
| nested half-normal | unbuilt | 41 | 0.0306 | 0.1396 | 0.3391 | no | 0 |
| nested half-normal | unbuilt | 42 | 0.0059 | 0.0803 | 0.3028 | yes | 0 |
| nested half-normal | unbuilt | 43 | 0.0070 | 0.0540 | 0.2094 | yes | 0 |
| nested half-normal | unbuilt | 44 | 0.0051 | 0.0755 | 0.2765 | yes | 0 |
| nested half-normal | unbuilt | 45 | 0.0054 | 0.0747 | 0.2991 | yes | 0 |
| nested half-normal | unbuilt | 46 | 0.0025 | 0.0270 | 0.1660 | yes | 0 |
| nested half-normal | unbuilt | 47 | 0.0050 | 0.0954 | 0.3125 | yes | 0 |
| nested half-normal | unbuilt | 48 | 0.0075 | 0.0815 | 0.2384 | yes | 0 |
| nested half-normal | unbuilt | 49 | 0.0029 | 0.0506 | 0.1710 | yes | 0 |
| nested half-normal | unbuilt | 50 | 0.0044 | 0.0502 | 0.2495 | yes | 0 |
| nested half-normal | unbuilt | 51 | 0.0033 | 0.0470 | 0.1672 | yes | 0 |
| nested half-normal | unbuilt | 52 | 0.0117 | 0.0775 | 0.2473 | yes | 0 |
| nested half-normal | unbuilt | 53 | 0.1057 | 0.2601 | 0.4682 | no | 0 |
| nested half-normal | unbuilt | 54 | 0.0058 | 0.0651 | 0.2652 | yes | 0 |
| nested half-normal | unbuilt | 55 | 0.0038 | 0.0457 | 0.1774 | yes | 0 |
| nested half-normal | unbuilt | 56 | 0.0056 | 0.0452 | 0.1583 | yes | 0 |
| nested half-normal | unbuilt | 57 | 0.0036 | 0.0405 | 0.1810 | yes | 0 |
| nested half-normal | unbuilt | 58 | 0.0099 | 0.0940 | 0.3604 | yes | 0 |
| nested half-normal | unbuilt | 59 | 0.0366 | 0.1463 | 0.3489 | no | 0 |
| nested half-normal | unbuilt | 60 | 0.0158 | 0.1067 | 0.2981 | yes | 0 |
| nested half-normal | unbuilt | 61 | 0.0051 | 0.0466 | 0.1838 | yes | 0 |
| nested half-normal | unbuilt | 62 | 0.0049 | 0.0642 | 0.2725 | yes | 0 |
| nested half-normal | unbuilt | 63 | 0.0092 | 0.0866 | 0.2509 | yes | 0 |
| nested half-normal | unbuilt | 64 | 0.0087 | 0.0657 | 0.2371 | yes | 0 |
| nested half-normal | unbuilt | 65 | 0.0016 | 0.0344 | 0.1690 | yes | 0 |
| nested half-normal | unbuilt | 66 | 0.0124 | 0.1079 | 0.2791 | yes | 0 |
| nested half-normal | unbuilt | 67 | 0.0009 | 0.0214 | 0.0983 | yes | 0 |
| nested half-normal | unbuilt | 68 | 0.0120 | 0.0726 | 0.2163 | yes | 0 |
| nested half-normal | unbuilt | 69 | 0.0043 | 0.0486 | 0.2785 | yes | 0 |
| nested half-normal | unbuilt | 70 | 0.0030 | 0.0428 | 0.1767 | yes | 0 |
| nested half-normal | unbuilt | 71 | 0.0030 | 0.0337 | 0.2007 | yes | 0 |
| nested half-normal | unbuilt | 72 | 0.0057 | 0.0777 | 0.2881 | yes | 0 |
| nested half-normal | unbuilt | 73 | 0.0020 | 0.0336 | 0.1589 | yes | 0 |
| nested half-normal | unbuilt | 74 | 0.0018 | 0.0295 | 0.1694 | yes | 0 |
| nested half-normal | unbuilt | 75 | 0.0142 | 0.0863 | 0.2367 | yes | 0 |
| nested half-normal | unbuilt | 76 | 0.0033 | 0.0466 | 0.2006 | yes | 0 |
| nested half-normal | unbuilt | 77 | 0.0131 | 0.0973 | 0.3570 | yes | 0 |
| nested half-normal | unbuilt | 78 | 0.0095 | 0.0742 | 0.1979 | yes | 0 |
| nested half-normal | unbuilt | 79 | 0.0095 | 0.0712 | 0.1725 | yes | 0 |
| nested half-normal | unbuilt | 80 | 0.0020 | 0.0245 | 0.1098 | yes | 0 |
| nested half-normal | unbuilt | 81 | 0.0026 | 0.0382 | 0.1638 | yes | 0 |
| nested half-normal | unbuilt | 82 | 0.0069 | 0.0868 | 0.3034 | yes | 0 |
| nested half-normal | unbuilt | 83 | 0.0216 | 0.1368 | 0.3657 | no | 0 |
| nested half-normal | unbuilt | 84 | 0.0045 | 0.0481 | 0.2334 | yes | 0 |
| nested half-normal | unbuilt | 85 | 0.0091 | 0.1211 | 0.2828 | yes | 0 |
| nested half-normal | unbuilt | 86 | 0.1156 | 0.2379 | 0.3941 | no | 0 |
| nested half-normal | unbuilt | 87 | 0.0037 | 0.0404 | 0.1535 | yes | 0 |
| nested half-normal | unbuilt | 88 | 0.0021 | 0.0365 | 0.1657 | yes | 0 |
| nested half-normal | unbuilt | 89 | 0.0366 | 0.2036 | 0.4212 | no | 0 |
| nested half-normal | unbuilt | 90 | 0.0055 | 0.0719 | 0.2756 | yes | 0 |
| nested half-normal | unbuilt | 91 | 0.0023 | 0.0349 | 0.1339 | yes | 0 |
| nested half-normal | unbuilt | 92 | 0.0081 | 0.0626 | 0.2250 | yes | 0 |
| nested half-normal | unbuilt | 93 | 0.0048 | 0.0743 | 0.2382 | yes | 0 |
| nested half-normal | unbuilt | 94 | 0.0138 | 0.0873 | 0.3024 | yes | 0 |
| nested half-normal | unbuilt | 95 | 0.0121 | 0.0882 | 0.2501 | yes | 0 |
| nested half-normal | unbuilt | 96 | 0.0117 | 0.1144 | 0.2962 | yes | 0 |
| nested half-normal | unbuilt | 97 | 0.0043 | 0.0671 | 0.2067 | yes | 0 |
| nested half-normal | unbuilt | 98 | 0.0058 | 0.0550 | 0.2284 | yes | 0 |
| nested half-normal | unbuilt | 99 | 0.0141 | 0.0795 | 0.2625 | yes | 0 |

- nested half-normal, unread: covers 0 in 18 of 20 worlds, 0.90 (SE 0.067; 95 % Clopper-Pearson [0.683, 0.988])
- nested half-normal, unbuilt: covers 0 in 87 of 100 worlds, 0.87 (SE 0.034; 95 % Clopper-Pearson [0.788, 0.929])
