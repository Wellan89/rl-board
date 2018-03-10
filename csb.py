import base64
import math
import sys
import time

import numpy as np


MODEL_DATA = {"alpha/b": [[6], "XTTNsDEqCygnLGgy\n"], "beta/W": [[64, 6], "6DKSNPypUrCLMCUtIzTpOVC1yKn5Kiq0eKMTNdssbbFwNNYgp7rRtc0xCawLMo4gUi1ruCg1/yxr\nNM0rIbVMLJKyS6+JK98u+CH1tu8rCbHQra6hkTkprNQury2DrtaupjokuhI5HjT6sPqwxS8HsIgy\nsDLKMogvtjXSOGSwiy0IqVqwP6mOq4kzvTLFM94pgy8ct6gtjLDGsQi1bzDwM0Gypq5FsEOqPbK0\nt1qwvzJwKzGybziALeczK7FenYIwCTaMNRaqrrJdIHUv4jVnOLmzWy/krYkvJbi2uMqpCTH8pm8u\n9y4LNfGtCSgWsistFK01PRy5qK+wsOmkajy0s6Uz7657regsjzh5tDmpaDXTrzapmjS6p0M2oDAA\ntTkwlLZyrFKsqKvBMhI4GTbUN+Iyqq1Zseqt+TXrtT0v1jB5Jd0vEit4tTKw9DTjLN2qA7Xwuekx\nBC5GplueJDYpNEItxCXkq3Cw7q7jM7m25zBrK0qtNLYCtVIoNy3CMks0GDXHNxgwUTEaKG4j/TcL\nNj8uYrGctP2xYbmps7ev060UqYezpTj6MTKrP6dJjrWzWDB3MmyrPqg6uJC2ZDjarxIxlbIkpDIt\ny7QlLl+0k5karomsli66rZU0b7JHpYUr2DnUNP4xz69JrCIwEqA2N8ioeTY+I5awKDl3MRAtQzBu\nL+ixKLW8u0c0N7Eds4Uc2TRlJLisfi0stGCgzCYJp2SsFp1Upluh2rN5nTSqGamwNNYwBLCTt4ew\nkqZztbItFrZkuikqCbAvKsIwPbweuHiqPqmYrUC0ZTRkOJYx1qvBGkQuhDAgr280njLkL5UrAjYo\nNZYwiLMjrmAyrR5YOTe2uLFcNOKr17gFvDMzjTBNpi0wMbg+vIQ1qi6lsN0vt7PfunU1VjCINcEo\noKq7uMk1x7L1s34yda9TuBs4s69usDUwbTYLoResFaymspOxRLP9OK8rLrOyKREuBC8nuVY1prCo\nqzI3n4wvsAWwxrDFK4CRyLEUndswPawsNL4s\n"], "alpha/W": [[64, 6], "yKzbrisqGLGapaurUSpJuJIvtaPCsSYsty0ZLnE1Eq8yMrKcDzgnKbGwrbFBs1wwvipjOx+uO7Ef\ntCCrj6Vgs+glLKcztO61a7SaNq2l/B0wNHosd7F0NGIxtzAXrYAzELaqPCKtZ6pjMiAzwjT6MbQo\nZarhtVIyX61wt3QyUS5GJsqhtzGZMNqqQy8mNIqxaK2zNIwyti2ToVovXrP8st80MLTYGCqwvqwK\nqSuxuqpzLHsr6a8FNZii56UcJHWwwSgGs6kyLbR+sS8u2rHZtWsxtaMetcawErI1NxumUDTuNKit\noK7JrGCm6LOTrhMxk7F8vT0397KMr2mw17eEN6cMdK1FIQcvWbHtOPgiPLUwtDmlobV3M86pk7AP\nK7YoziYaMsmxxy0MqIaiVCSMrhssb6tYMiMksKsGObshAC6NpauxKqHxNAgUkbQ7rXw0GTGYOiMs\n47NsiaSvR7J/MUM3PLNmtAY3UbFnufyo8C1gsqag0rDDsmmxzCrXsGiz6jGjMGM0EK7/sQw1c7Ep\noCY0vZ4+Nq+q3jTVMGKpIK5ks/S0tSxTMV4iBq5nsiSx5TbNLbUtCbBmLHIwO68gNa6sjLPAJN6v\n/jRpsOElkqnyrWMczC5HrNgmizRgNkGx/rIop6w0SzBSsKiypSPvt5MmB7QhogY0rqAbMl4xDSRq\nsN0yYqpqOgesyqats/WozbArNF0oFKoftCO1xzVeLBcmrSlSJD81VrbZs72w4y8WspWzwbZiM/wd\n9y3uqfYvADF+OZG2GCzlMWIl7jhcOMmpEK2UoAwvBLEOuEMw/C8lKoKej7GcNG+q7LFcs6Ew7zLa\nMBEcYS0osVivgLUouDozkTQfMWutyTQKO0mwubIDMBQvHTgSO7iyGLSdshyy6jQ6O0O4VyzErLIx\nVjUcOyIopzH4sLSvTTJGPA6ppDCgLBKwxK3zM5g1XbB1MIQzMCyZt4osVbA7LG8xpqH9N1kxFyuM\ns7myoa7Lse+s16scLMcv4C9pNEgqcDK/raSp\n"], "dense1/W": [[64, 64], "taw7rtavwB7KrcEvhKc0pKkpfiqLJS8tN7DGKziwqzLPqWMwu6DZK6GtNCwoIoQqKKzpLis0QrTp\nrsq11rHooHOydy0EsGKp1yueLwGtzKyBrQsswylEolKiL67/hnsmxq4ysBAmFSTLp9+mXLCBHJak\nezRMm8ygvbBaKnouUa6UqH+zKyoHsLOrOp6+MQW0eLhJpuur4LMHNEGxBrAOsEoflSOKMZCmSbVD\ntVuzBbVIMWsuCbPTraQwiTAntFKubqRbMPQxYayppOasnzFpKgykQrCbqqIwLjTTM3+wyjL+LrSu\nvyvBrA8yhqoAqjInNzGyMZsyC6QzICMs+K1qLJIwKDcJKkOv5Lacrnezm6tDM+uvaTRHsfG0CjVz\nqG8qZTQrNKOvsTAJOagv96RboiuylC9tpyWw+LdxrEA0hy1CMbkvAqWmJYOopZj3sQ2vk6wKNtum\nvrb2qNmluy87sa239LcXNgizfilMNZq3tLfPuGeubrjhrnk007HaMVGzXxtYKZ+tp6y6sSY0wrCX\nrSau6S3KIfirfykXJnce3qyargQwIyfILQ0kECzdJB6lmKihLOAqr7MCrLsvUawbq58liTAkMmwq\nuyaDMMavbDSHM3Wx1bPHqNwwJqymKnWsL7Otqr0jsK4DMXOtLKX8MK4tBjFRMdaqpyhiNaCwGaca\nKxgqFS7GLB41iiTgrT2gp611KEytLzAEqV+tdKdQMf+qhaRitfcs2y0vLbcU/ys4rgAx/7JOJQgu\nHjO5nFQrfCz+Ky4wwi1lIcU1srDeqZAsqi+mMYAowi8DpqWwh7AJr2qtQDBeMAmsDK1wMAcu9aYG\nsq4sqC7RJrSseKFEqTGsuiUOLWcvKjUVrWOxCq3nr/8yHylHMmuvpaWFGkAp5yhAMZ+rva4yMAww\nuikNLsMtrLHKGZMs+S/bL2wsxCkJMYazPrEhsEexOiDhLHkrprACLWuomywnre0tdavrsGyssbCr\nrnAx5Ku7sHwuSjClrGeu77CwKjquJjS3sz4xZzCNphatBzJvJxuxKixFKcAnJCuBsj0oBbH0Mg8o\nsjEZM8EubqkoMwEfvyw8KH2pzBrRLMutWyryKzYxCa2TpiKsFynsrlYoDSUfMralxCz3Lj2nwrE6\nK1IxtKo9qgYq0S3ZrfOpKK+IpRqUaCjHrPqre6xJJMoovy69Jw8yCLAmNL2sryretX8oCjJLrXYw\ntbSjsy+sNK+rMeos/aVMIzc0/C1gsYEYCq3HsmCy3rTws6aqXq5iNGmrbippJditaCriKRqoFylb\nqRMsjafTtMsvgrWkrakmhymsspAv8zHspdMqa7FfKiCwfq3XrqOoUC+KsAAu0DGCrNQvMSmXsoYp\nprCsLkSnszDjoSuvcTD+s56w5LQNsGmlP6WZsPwuljFSrlwwgqlRsTmpWrCQMd0piKuZKo4xlSzd\nMcCyMbJMMawxz5vPLIE1nioRLQCyBjSnqPcoA6yWMAgps7ILLoEs+i/ULPEkhi/rsDwrCzWoqv6u\nYTHuMRss0jGpMGWxZyvxJiwsLa7+p/Up1qpvKreuvK+WHNcnrSmkrAWpgjFXMKewZyfkrAgrKC1P\nHO6sUjSLsWku8zDFr52zlK0qMN8jV60Hjh+gKbGsIo+lNig1pCswPyn3LyMxyLHtLn6tJiuuqOEs\n+ah1KlMudiQJqXMt8K9glHCtFbE5NjSrdjHHqkakOq3FMgmwCDEPLh2vPLJzpD6teClFpxgsoi+q\npOWR9ikTMsmozKSYp9Sz3ylrMQurRiuFregnj61/sVYwiTEaKDEyC7OGnJAp/TPVLJyv4q4bLJ0t\nZ6zHLg6wWKkbKDMxbahQsGiy2qnFLjEwbq4KqPEqBLAtLmghgi4xLRWoBzBpqI+nqCnjMEmu/xxK\nssoujaa8qWsocag0IEKu1C6KJY8trS7SrrWzwK9msSAsFaF1ssOtZiibLb4yGjBfrEaumS1Lqo0u\nD7BEptutMS8MsDa0JiptM/guHCFbK2QrVStFsLqv1DJGL6+wFDCCrzslDi/XsKYhP6YsLKiuijIM\nODo1yK09udE0O7dBuAO8ArGMNeaueiNXM6ewGLYiMKEu0LUiKJw8gbrnuNm2YymiMWK4FLWXuGCf\nezRwIiQpwCmhMBy0kSyeuJI3zq3TLOC0kbRWueMflp/vMVmxIrjyMIYzjbZGKxE42LYOuAm5x7do\nMqiu7TZsubYiEyi0LGSqYyaBpt+khhganjAvpC3RqiIoEq3bJIOYYixfLQMnGbMPrCsqXCbil5ml\n6CwerjqowyDuL2guGCY6MNCw669IrzKsUq00rvAk8bABNm8qnjFuKfGvwB+uLPApGShZL/2tAC3b\nraapVKnxKMAsgSjbsgi065gsmWcqKjQRqNEw9agCozYz7SCrpwsrua0kKWyhSCuDKkwvzKcFn24r\n5zADJVirfClzruOsaLE3K0WoMCoDsGmg36Rvq4knOiN8Iyis7i7Tr6EtACN5LRomlavYKSEwrpe4\nplCk663dLBwwbZMvrLasoKkpriMtpascqmK06i8cqGSsk68+KYIxYCrxsK8tRiinrRAuwrBrs6Wt\n2q8qrmgwWKQSrJgsZaYAM9ewGZuFp+gxD7EGs0eorhumscesA6wZKnoyH6nwIuSwurG/MWmr2J74\nM4msKKkHqbclTCrknKOoG6twHSMs4C5mMQusUa7CK34ppDAPMP2oGSTpKrEuDK3DJ2StgKxyLaer\nu7ApI8uuFq4Aq4srMDNeMVudeaZqMOklhyp1LuMlGTCIsgil+yXVomowrbCpJbUs/K60qOKtnzL6\nKD6uhS/kMJ8weC3grNQs9KW6KZczGbElLAksshh0NFcwUadpHAqsvyU3qPQycSBRn4Ou+KnqMmGt\nfiCEnpAwYTJILymqHqpoqegrojMUNEWvIS0qrdAwlKyaLxCvVag8LtcsSqRgMaifth6PKnoxRCwY\nK8Qtf664KB6ehSw+s1ovOTQirNq0zqGHoUKvKiowrj0rt7AcLPgodbMTL1qziDHmKi8rtSpHoTOj\n1angJuomlyAaMYWwzoySsnKmzaqHKCmtSSrcqF2q361brrMemKOVpV61K7CorycvRTIQNCgv9i5l\nqYUyHCzuJCkeKbGBregl9imkro0o0C6HMqimwKXpokQsKzQRJBK1kzP/KCMvVyWDrMMuuamHrfQt\nE7A8qoQsHLEJNFKzty95rWEvXiiuI9GqOKkmqmszTarzo2wsEa4YqSMuXShlrF2quSbBrRmqPi3q\nLAosW6xypaIc5C51rgokWK7hpAchZjFrKTY1PCsostQpRizXsKQrZTCxKKuxSzCtMLqsSiRVMqGq\nHS4rK82vAiwkLlStxjOWLgizK6y2sSyk3SpbLPauXiScsqQzYTDerC+ooauaIeCwvykCr7+k/LLk\nsEO2+TU6r7AyvzKbN/80Y7U/Leww3rH4pdoz66sish81YrOPt48xvDSrmYUujrA8OQIw6DZJMHK4\ntywopykiRTb3LPMsWSttGRKq5TG9uBOtcDj/NS6uc6RqNEQ2HzZBtQWuEidDtA452Th6Nq46HzkU\nLyK2Qjc5rSwxtCj6KNIsqbNaKOCva6QwqMandi/RGviymagmLXsaz6r5roupya61rgMkBq17r3cx\nyqw0o32xz61grYE06S0cLJUwMKyfpkosCzRsr3Wsb6qEsJys4Cv4KNMh4i3YppctSiMfJR2loirF\nq1mvJSwNLeMhx6ylKeWotikxpcqsn7Bfq7Qy5jGwryMqySLEJy8fcitCrP0w6h3uLFerqh8jrf2u\n5hinsQsnTrAWrN2wnTKdqA0wB7DRst6rXjK0soyq0q3TKsOyrZhqMUCv/iaOLwkybrPiLbIxxjG5\nq2GwqLFnLgywoTBhrWQwyRtCGw2tp6bhIKgvkjIqr2eoaDIOGJEp9TBhpkChQS8jrrIkxDIRsAQw\nky96Hg4tV48SsCErKZ4Vq8knkSwWrvWtMqq+LYyaJbGvsoOr86FWNLGq/C7oKZSwcSkqpzK0OKfl\nq4UuuaQWL+ivwSvzq4Ek2LHvLzG0CrD3MNUswyfCrWcrZzAZIc2uMqVAL+4snTI+K2mobzOnOEyj\nALtmtwczM7RvmSezDa9HOKis3LGXMwG1zDCINa03f7XCLvQ7ZaBorgcwdqdIMoGt3rSSuE+iMDKT\nIQizVTTUsjssALPQqvCrfCofN30pZbGbucUwLLT+MFG0mbiwupY49K/iJgE5QbrQu3C72LFrt0ss\njDdFsVSv26+VM6Wu0DAHL9cx9a9tro6ptClUM0WnRCjBHv6sAS3cLDEosyPppM8vgyYhpNMkG6zC\nLEkSByaEqPouBqs4sCcunC/4rCKxxqGyMHyoSLFCsN0b2iUYE7WwMa8HNHShULFCqHKuEac3qzge\nkqmDLSYw0DJGma+uTy1WJr4jMy2XrgyveS5SpVKoFjAgrmkz/yrasMUmaRx4MRsi2K4mogixrbM9\nJ2kwVqW+p5qp1LNtr92tyy50pnGvTR4XISentrCarkotEbCprhUzU6X7LGipxq5qpFAoWa3TrMov\nmLMaM9woVp8/KpOxSq4HrrIpNbEyKAkmzzCfG34tEqJNMmuxdK7WLNItTqULrgGuRa4GLg0vja9e\nq/UwlDFRoJIjG7AiLaSxJqVGrOEptzA8L5MtCbLPLa4p6i1dpVwrsK7UsUYoDSaMLRYrezVnq3op\nci3iLfso5iI7qLukIyh7rKMlri+9MM8nXa1qMROpiS0srnyy1ywaLv8wDyZ7q5ody7CBL7WvKSOT\nMPsrWixZLvMjK6kWsletPKTTq9cr3CR7rsOukCt0Msuv2apEKsWx9bDFMOksbaDGroqsGCUVsXqn\n1S3rLEIuc67HtM+r3rAhNKMxk6tAq0asShQ6s8wtGDAQJMWlnzDWJhowlC0MrM+voqi2sVkrkKYy\nrmUxmqgwLTQs0rIVqqkyMqZuI56uPLA5MGiwoLCmrj+wtR40LhOu/C2BprOugS/TLF+uO6hwMpEu\nuTLRrxMxd6SEqB6u1y26skaswaPQriqsgChnMUiz27GRNMOwi67UslO006woMI6tYSSbLvUfnqeQ\ns3YwmymZq6KoxqzUtUwnSCvEpZWy/C2zpcqtd6yTMO8whCgor4+m9K+/sNsxXLF/sFAiqrCtr+er\nLDHHmHKx8rHQsWm0lTNRrfAwGKQVK/uyBZd4MFysNKwDJOmxhqxrqI0qTqR9siOwLqhFJxat2qvM\nrLmdmjCirqQo6q7crV+qUi0uMDslAiyqLQyvzbJ8rTKlzixypaakEyU4qbUl5Kvuq2YkciWhrGcu\n+yyXprkpuTRpr7qoc63OrPewPirsK8SkuCNIs2ijxqxBMDkoth1qsBS1CSCAKcIq4i/xpmije606\nLjuwrywDMoOsKCsCMbaxPbDYqT4eR6Q9HmCwbKzVKoUvHigEsw0scCgfpQywfKIcM9QpLiwFrqEs\nXC7VLpWuLbG7r8yuyKnNsYIsNiYSKbKtuK7YLpmsAzI2K1ItNjInLOIiYa7/KfwqrCgstJwbIi8/\nqp2vMiaoqHwveaQXq0qlALLhqm0vUSqHKoksSaz+qYIw6yXfrB8viKnLoN8qIawCmmuN5bE9sKaj\n8yp7LeUsQzCSLuIp8B7FL12r76yJKC8x9TM0rzAwAbCbsdEogqjbK/Svea7LsX6k5CP5qeK01Cos\nr28oD6qkqiysR51BMmGpprI2rQSwhK+eKRMxHS4xIe2oQbBzremvZa0bm8SyGSwSsrSpNawJom0s\nSK6YqOAvTC9or6SsOqhmpR0uvCwEqCazuQrpLKkvcSnFH/quD7PhrUivpi6RsosqdiF7o42oa6ze\nJEGpvKgbqnuvzDF6LaAsBagZq74cFLIysy8shbOZqaElta0mIrSucqw2rhutP7LQtGY0RCR6rCsy\noSb7MjarNixuqROcubTvM6sznyx0Mr8v4bEiL4o1M6OUrmItLCj+qiIwQimoFx+vTahRIHWoYChM\nNHS0a7R2s9I0ICsUsiusN6gtLSGkpy5EMxIzDiq5Me6vGiIuMNK0RiwdMG2mbrBhLaK07jCKr50w\n75/3AXwsoK6ar1gkojIssnMsqS0CMK0wFDPYpdayMa6RLn41Pa0wrEoyvp/IrgIwvyrNrkInba8Z\nJlMrty44KQoq7St1rtGvZC7zMeCnKCj4IA2jmimtLRyn/C1QKk8vFa+IKASqRqnBsEwzUCsmJuIV\nMiWKquIyByFUMJ2prS1JLFSrCC4ALiCzbaTWrMIyqy4NsIatWq9FMhIwYKXBo2mtV62crNosPChY\nMVGiWrCFqQ4py7HeqZOwzpF8LOykUS+PIxYy8q3PJ/gs5i5ZFcewp6dPLjauxCu8o0il7bIeK9Kh\nKDGoLR6tGLI+qpoWvLUUJiMwwi5dqQcj3Sc4prsYirGMLPeo5ZknIRIpiC24Jm6x3ix8pWkpRTBr\np0uyLq5DMPewvS1DsgkoiyZCKpQizDAIsYIkfSm+Lpmlei7tK/OoWy7YMTgsyLPGqTMpG670rG+w\nSK8DsKYpGKGELkSy5y/LqU4tWartry4uG6Xtseop6SpgJ84xG6leqAMsb6x2MfIiyy3sqiuuyB49\nrJmhzK/fJUwk2qkUrbulHavGLWuwRbAOtLGulJQZqgCvfqWTqxupGbCtJHWtKCv+LOep6hKdsYOu\nTqhkrParfRXLs9otLifHpIQwM6zhsYqqwzEpLNuZjKyYsDiudKlOK4quAC/TGjWukzA+LnKuEq99\nMKChV6ResDMnwqbPLVMw2S7aKx6mWixsro8wly7nDP4t8CeNocgx5yrNp9SjbysNqf2sVK3SrRWc\nb7Hvq/cuTC35MCCZVCRCMD6uBDLzHF+ww6sbrgIwA6RLKpUv8i9up7YwUTA1NgGz5S5DKS+fEzTW\nrdyxMa9Fp9Oyzapgq4YqVKqDMv0utSROLF8sUR1fLpmx5DOkNPqn56/LqfehTSlar6muSDA+KPau\ngrSlswqv7DOdJ3YsJpZwp5iuT6RtoCCtySX5LjQv7q3upqmupii5LUis6LAjpW+uTp2GrY6wAbDK\npmAjdDANLT0olCjrKL6wczQSHemkAym1pdGwJjQCpk6oOqw9Lm6xiCUKtGuxNy2Dp8Uu5aLiq1Iw\nvzJNJjOujyUorCisOKnKslmoJa4sriKwEqiprd4ucZ+aMEuvB7JeL+GYQyyZLcIxayDLpGAxUzGH\nrgIuI641JL+s4h/jKsSlWS5GLqGsuK3FLFyzAC0XLU0sjysTMSMwwqCQLXMgVa88Ki8vRq01LV2b\nALDwrfYpDqt0r0ycqKeapAisLixALCul/rRvqvacl6ohMowmhKIzrmWxEan2sO4wSTQELx0p96wZ\nq/wWubRXJgo0dzTMtHGgpy05MwswIS0qIB0zoy9AGD6xIC05LR2krDXRLiOxfi+iLucjvya+qj4y\nzCZwM00oGDGfM9qwR7KzJmKxcSRnotAw1K12kcaw6qxHrzUr7Km6Mqqqm7AjJfep4iTJsDEwEapR\nLngwmyw8rQi0yqszsA0uBTBOsNAwNSDSKW+unamEoFOp/rZkpAynqqtWr/+vJawdLfon86Q+rX4h\nZS9TsT2sn6/7L8crvxKxsbmwwDBVJeCtCS9FqJgvxqRzLiurvqjmKgmsYiLAL6GeSSt+LpYl/S6I\nKDgwGydcqGC0da01MK2vr52TNLSktirsHSawzqsftO2p8qiaMN2ouR1zLxMtHKXJsRwka7MdpSEw\nHCHJr7iahy0TmvSpFK0fr0ExGSr1nfaoEqkhLPyxWyhhqrQeK6wjMcsuPSTQMRiqOKkYImYsr7LJ\nLJKU6ypQNB4z+ybCrIWqna7dp26nJrCWMbYw+iqcsPeqlacVNd0pTKLZqCYx7Kt1rSMvjzS0qKgz\nDzLdoFQxYy7RL7avnDAlKI8m5Sm4MWEi9aojtKEskinepQAniTEqsJoexyiopr6cUC5HrakwOSm/\nsDio4imRJvIo56nrkQsyJan2MYwqfLHdrQwzNq2vLq0sEDRbNKMtzrB4qyWsTCCYoTsv8CrHsPOx\nOLAmsnEm6izPLCYpwzKWKnownic8ITEsNCoaKbot8iwYLb2q/LAwMgEud69WL8Sq7DGdME0j1it/\nJNColq4pL7mula1Lsmm0RKorrJcofS0NLMyveK4GqQgovJeWJFEj5LRvLhyvRyhPNNku/LI0sgYo\nBKZ9Kx6xYDChMGqsjp6rJXOZbhjJMs4sxZqrJVywvyiELG8wFKMdsKiurC2CqiIuf64XLcIpc7Dn\nK8atOLSIMtSoQLJcKEAyXKF3qH+vsKtjpWwi86ygtZerxhCLrNMnKqsTLZkr5LECsFqmfbEQIc2t\n+CqdrIulpzBuqyuuga4sKGEk/KiFrUcqATJOMjOpAZ9BNEaqd6vGKMixc6EtLDCyQCVUrvEtwKpI\npRku3R/AsWmwnKvFK5SsiKz4Hq4pk7C5MAGrnawGLDmwParVKySy7anwJZivi63FsKms3jAjJlyu\nMinKrccpyqqmKYgny6p9LRqu+qkqKyAxgqiyIZakkyj7riQr6as6KYuhALCirS4rfZ0oMUmwr7GH\nLVapibXlsQQsvLOMrPexrCaNsWMywi1kqRWvaan9rvsjRi43qbIquKlYsLssway9rmgs+TGBKgMk\n5avcLbObpDErH3kllytwpbAy5y1HtZGgFDAaJh8uLbEOrDEslTR9NUgwnazAt2y3py/1tIUxPzRy\nsu0uaLXmtSW4yDfmsFu5lbrQtDu0yTRqtC+zCC1cN8qvurDqlDKyTrbaOCi1VqwstbswwrBgtRm1\nw7UhN6mpWiyBLuC0aDgLOjC1syhwrWW3MzoWOg84MzJQOeSrYC2zM6+yBjCZMG6p/SlKKRCy7CyI\npLEpSDCuLO4sZ7JOqYWqdKoRMBcswS1usgQkMxxPrFkvAZ2ZLjcsLiZrMesXIrCSsR6pATREsGmr\nki8PLPsn27FHsNOlh60cJNOoj7BBrS0wKCUCrICqA6yiMMAswa7gGM2otyLNLzkl4S8qMW4yOqU6\nK9Aa3rDKKrUjOCwOshkxIS71nBmrkK8DMDyltDZapFQtKixNoFCvtaM5rV4oLqlWMrWwKTEStFur\nSiymLfOtq6YcIEkxTiDPre2tE53HskqTvDAKsJ0mXyWnrRGwyK8Fsssuzq6BMFAswDMSrFerniMa\nsKSvKDThILqsRDJjJS8waqUwquGs4C1vLSowmbDGJWCrDy7vqFoquiruLgCnTqy+KI2qhKURqyCq\n+qw7rAiykyj0plStPi4CL3ycCjC/HKanUqEwKWIyUCl8MTcwDioPrMCsNyi+sMosMy7mJviul6Tv\nnDYqOjBrNEmt1K2wLA4s8i17rNgtgDKRL9MUGyyWJqUwmjP8sSmq060dM9OpDaGhtJsv1a+eqQyp\nUyaSICgtvK9LLh0tPCzYLSQrBS71LMWpoqxdsCGq+i14qduxsK6cqlmwILLNqN4sgbBEL0WwySlT\nprACXK9cJucDfylKLqqxYqdZMnqwwqhaNCYoOau1p6WvVTOsLwssRpY6sVOloy/OqG4tga+eHV2v\nc6xusBgylrGHs6MvraYTMbWrwasNsJKgiaxIquesCi/xsaktTLATKlo0Aq2WruAUDzTysHSytp8W\nMZan8aT1I8yr3K/nJAmsbC5GGaIuADAgKxGuJaowKJMxBLPVrlcUcy8lLBCwwq8GoOUtHK98NdWo\nLTHVKns2DK3XrCUqdKvrs2evpCQDpRsqVSTHsdysuDE9l66yHyygIlgpeK42tA8sARs2K5klCrDi\nrfKvzKxEKk+yWbDyqvGu4rFXHjmY9S8QMIuyMq8BpFguiyT6mmO04isopFilRa7HJRynjKKILr81\niagZKbEhGyxUrQGkaa2goLMsqzC3LPslX7BdoMSYnih1LGSwyzGIsf0kFauUpC2y3StlpFkirC2Q\nJCCwmyUfMtutQCaMKxCjya/CJV0xGKQSsAYfuamrJHIsK6WVqwOmXTO9Jqqq/zA1LVwySDASLvex\nXC1yLD0qcy6ZLhQtTyEOK66rMJn5o/OnpKo2LR4pDy+uqSmrujAyMaOwUS7GqWOt9qjTLFysTyzH\nLJwtx6VfqDCYxqURLN2sr62grDewfKEZIk2tgq3eMcWtpS8Yrdig2C34sV6sZSjlrX6wrixFr3an\nHDP6M/Gy6iw4ItGxTq9pKd8wG6RDpemhqym8LHimFaCLrT6dIyl/MAQwtS1grEusGi9/KYSuTqaS\nrTAkQrI1MI0tr6qXrS2uMCwzpgG0+DH2Mr6ruSoNL7YsVa3DKHiuvLFwqnMsVLLLrg8s1B8Pspys\nTTLTLPcrwCvvrIcxDbE4MaSuprAyMfes/adEL4Moo6WxMPuom7C5rmShvTIBqZqntyTFKAkpKi2C\nskOwg6qIMU2cxKgsJEEtdC3sr58rPjBCLo8sdDFyrIMshKtlsomtJ63WMAishKqGrdqtVid5LY4w\n3K8HGHUxkSpzLbuy8q/4LB4NdCyHruyu/jEYLzeqHa0TqZ6wvi88tn6iQK1kKHqw4qwesPGvXi/K\nLMmvPK5GpxiojDARonYv7zOVMPKvFSpksckr55ruL5YtMCFsqTYkPyhzr4sxnyu5K0mzeTAPLhc0\n7q1BJTYtQakZMggtXLRiqIinpbBSMZ4yR6toKOktoDI2J3mrKTDwppYzUbACLj0mH7L3qjsn4Siw\nK0Soii+lqEAnISWRJh0sVzAtMZ0uU7HIrC0kH6jwrFcoXbGksE2t/CwHoyCxfit9JyWzWTN4rCIv\nIKwkJ+8yyySfMAqjs59Zlk6jP60DpPmxHi4yMbWqnqt6rNAw/S6hnq2u27EHoCEiQ7MWr6mpkyXg\nLRGwja+tMo8a6qeIJTyy+6ous0AwKiyUsjgod7INrN0yMrQgsSUvK6o=\n"], "dense0/b": [[64], "i6SxJnCs6Sh7Hywjxx2gJv0mWKQlnheaoyChm4chfScKn8aiMiGHJJ4sq56+JFShMCDvJJokjBV9\nmLql/6SiHGAh46TAnkGlZyChGeKhBaTRHzqTQBqZJHKfPqEzpJoeN6WplC+lYSlsnWQivBnmo5ek\nrhWWo8cg4RvnHBogkJ4=\n"], "dense1/b": [[64], "tiRdLpoZzayLrx6tnqoIKUwtLSlAK7KdLhwjq0SfG63oK3ojqarvKV8mlCozK6QqMq75J4SrmyLs\nq+EmkymBrtQt0izKrr4qKC+goCchWirYJvGp7C0xrn4sGC/8sLCta6uHKDYnSqBHIwgsUqpCsFKg\nrao8r34qHSxCsgWqKKk=\n"], "dense0/W": [[73, 64], "YakFsIKvchQerfUzOyxlKvEoUK0jtEKpkK08Mlqw3y67LL2kK6xLKk6ctJ7FMqauvqjsLokNRhMC\nrSewXbCvIzQqkLPcrNWyXaY3tJ4uSqWQMHYw6y9WqQ6w5TAxsxMyOKTdMDu0h7R/pwEtV63hKDCg\nryUJrVotpi8Srn4pFiQmOTK7izW6LoynKTXKrQq40LkyKse0Qbl6tQ0iALhRsG0zlTY/t8wW0iez\nt7qy7KoUtHU0ALd3N5o2ZiOTLT4iXSbYOMGv2zghs8A1RDBzLn8u/LSFsO4047XnrdEw1KWksoo1\niK77sCQwrLaQsG0vlTZ7uPmsPjIILH0mdDFltB6yf7QHsWG4FCYcNsQ0I7nDshm24zTdqTq22Den\nrkG13DGOLpm1xSlktoEty7XVN80t2Ddqq4u0wjQGrTgpECzVMx0qj7YJK6mvorOVMhO0mTJnNc2z\nDjSdtWS42bKDpM85yS5Psdsz37f9NJM3ODjnMbA4RCM0q5Yuvq7CLQy4SKAgrVWvZzEqshIoXbEI\nM6su2SuBOdcuMD+Mr0Ixnja7sSCcIDKAHo459KfDMJMxfq8zqBmwfaWnqdouGzN/MpMzRq4Ssne0\n2bBXqncp7CjApeA1lKhOpYQ19jHApPCwYrIgKzQ0VT5bMc4s2CyrsucznDBtKbcgF65Ftb2ochzA\nM1Ww+j11NyKyzKo7sDexyLBusOmlF7HCvRc0JLWEra8yXi5lr+wdXziWJogsTSVcQX6v47TYsEyk\nkbEupPgt6SwoMlQoaDfzpj8wsbFNrBqzU6tfscOo4a4xqp4uRBXtrKAvTasWOQkyK7KCJ2MhobEr\nrbUpfCvJsJEl6y9FqAG03TBkJj+3XKfLMkMm0rLoJdUwbp4ptiDALqMuNc8wf68foh8rOqOJwG2p\nq68YMkfAD6kNLjemqKv0pEateTFbM1EutS2QJycxUrPyMLmuOq9cLDoCyi7SrBMfz7DCreowkqTp\npuu/lTIBsiMjfykxrhKzHSQ/s+GwtjNCq3qwaDCFmUor/qqZsJK0Da6Sr4QiU6ijqGcxgy5Nqiew\nBalVM+yxWLMDM1Eg+asXLvspIqsPrI6prCgCJ24w0SksJsau0jCMMOos/SVRsI0p5awvr4uvBakF\nMSEhbCufraWtgCAjLZOvLy7yKxauSqTTrEIxbjCArbux7CmiMFymUrLyLvIu9bE4rAOwqzDxL7ug\n+CbdK8MtFqiPtpiYo6/eq5WnACwdKUgxci4RKYMfhTDKmq0rH63YqY6ltinlrRwr3iYQlWeuVzA9\nsMYSV7IhpNWo1JwZrjKkTq3ysKsryzAWsD0ooTFZLFClOyparz0qaqNsKjUjui2ZrUmnGi0fqxMw\nmS2trlowiyfLmEUuNippsU6q7bAoL6sphavoj9ExP65Os8wsuKlxqhqpvivSrHao3yBApBEx5C2c\nJawoCC7ZrjCzaDHfJ/Cpv637LeQssTGSMWcxGi2IqHkUZLDBMxexpqTWLaMqsSvaLt2uRywSIzqx\nF7GBrvktti3QItGpoiwCMKssPrAFsLwxPx2RKJyt6qwHI9MvMSxYqCgx2S2zLXEysqzOpDqmVSo8\nHAuri6gsMQwoxS7gqXkwL6xGsF21cy2rtA2rtjJ3LZmqaKpHMM6tjC2VI8wxaSjLnYcuG7PcNBku\n9SANsDMm8qn5L/Sp+CmBJNksgqCRLx+u4So6tCg60bTSMsu19yCstMI31zdkLxQz3DWzNEuxuS97\nMN8aRa77NeMpOJ0ptco3KzBZNDIqgDS4L+Y0urYlq084wrRcs0M00LdctUMprLWPqBE1bjHyMAM0\nUjdYs6S29i0Li/E3q7ZIMMio/jWZJPyuEqVeJXu0yTLWtJ+s/DE5MEaxrTSpLtsxyKLes8MamjQw\ntFC0CLblrrk00bZgrIym/rH7tIkmcys1NlOuuCHJrs6q4TDFr3yyrZxyr9izAybuqucx87JBLkYs\ndq4DryywMrCkL1Y0ijLtsVAyaJxKJ8OypCj4MaC1tjFPskQse7SEtRYtfS/YrKgwbKoYMJ+zC7bs\npK+Z4R7Fra+0TrWDLzIzoqxPrS8yrp6jLWM0vKWBru6rOqyrL6AkQi/5KmSwgaAxM3UXJ7VErWes\n3SdgKpM0+q7Ysh0iTySIJ9qsuiCmtHE0rzRaNAevmjKxK4Q1SS5etBK0upuxseCuEbDZtIGsgRyM\nsF4u6DT/Mh4x2y41skAybCJTqxOtlTJBLNOtTq/FrPuzhzR7og0vkxKlJxm2IDADn6UxeylFMpMr\nuKyqn9IxOzhNoHmz47SkrgIvG61zsWCxWSxCtBWvBTUhs3u0/bAbrTYwySZhL4UyPi+aIUcy1bAr\nrYI0iq34LNGyBrSytAcT9LdrBeowe6vmMLA2LKwHJbIhe7TlLPszPDEEK70YyTBJsDcheijCr2A0\nZiiRs0AddDZNno8yCZc9H+2lcjEFLBct1bQ6KEK3NbRtsR+1MbHhpok03rNCLYSxuhOVsbarajfT\nr5mwd56sMnstTrF9L10oKTIpH1Aw/TGsM4M2JKw2sTQrmLUVNFK0iCrgLdonwKpGOHK0OTXWMCgw\nC7XXIxqy+Jw0JtYwdTJuMKWwSiGoJ0olxC0PNIqqQKW3Lgm0CDWmKwerfbGZMtUuUy1AoC+vdjJ7\nNVcwMyhsN9Sf+bQmkaolj6xPuJ8vF6xuqVyokKcssP0zxBy7NQgoBjAutBEw9ixPr4Y0jrBorqMj\nqiA9tEuxySnCNhyuh6lfI6aqs66QnUinYrTYpis3SLb1NDGdPp0KNLAt6bENoPkvC670FjYwITPP\npVe0D7TOsHKwG6XZqBWoiDhYtKIlJzQ6MOGqwbRLNACyyyCTqkEwUTeiIFK0RagHMBsi/bY6NXEw\nsLJyqnexT6kes5SkU5p1owc0tjc8KF+0txTBow0qNx5CMhMbSDNxrQMthLS0M/UzFCW6GMKw16Ts\nthEQKrGtsA+wZzS4sx6qkCrTtWOuTalzqO21H60AMUew6yVruC22kabqtB60oqwVp1GnbjWKNGac\nUDHLKBa20yxFMxOrjbTZNpQ16LX0qeO0zKtZLxcY0qVjLQwm46ZMLbAspycWL1Kpu6DsLeQwnqU0\nIaUnayi4qhYoDybHqHsr1SIoqFu0oqQQrIMxKxwbKfSh35jtMAswpTF2MDSoY6fOqu+s9S3wMJa1\nkKp0oZ2u8SbfrqAr8qTyMLaxoSp9Jbssa6rFK9ouLazqp2KyoqhbLKIjAZqQIVixD6+dKDOqC6kb\nL8Cly6I3I90m7q6kKYgoLTEdMxizShrmIGCpq6yFJw6xnytosIazipmCq2UxYqrjrSGsUbE6I74u\nm6zxMSMrobBLHf8r3qjRpH0ss7ENrASxhbB5GXYsv6WWrF4iE7CwK1kxdzFRJu4p2y0RNKYw0qYo\nJCYtl6UsqCAgzzEhLNCkJC6rp4ckDSkIsa+lDDGAreisDyhIo56d4C6Iri8eLa/UJaktwi4RrJys\n060IMUEvhbCfqrwrXi/wI6g0PqyBru6xmbFcqEMz1bGyJdOuL6l/KN8eD6/KKTGwdDDlr+awHa89\nMFQtjTDpM2Ks0B3JKMgqsK4WqQKo4ap8sKAf9xwONHqnMi6XpTgl86N8JIGrWa7mMGexuKpAqKev\nyBqusUupgrA1pZcxz5k9qUMtjTHdGEGnP653LREwwyj8rNAsobReLQatfS+PrQkvsrL6rOwoKjLg\nMGielDSVJO4eZisdLQUy8SuiHCWeHrVyrbsuuyuFL32o+y3hL36zVSpaHuYsJyv8LTk0MaGYJBqp\nQSSdMFCsQqrLLOOqRjEvLe6x8zCvLjIu8q0NL+qovKBFKNIrNDKwr5usaicHJUctHCo8J6qhVZFW\nsMmjbKpjHpKw4CWmLzCx3CMapCwvna33JJyoV670qy0udSCUrzSw5TAEsuYkfq9WMCWcyq7TKoIo\n5i+Rr3yXQiomp2omgiuMKeMmgjAnLHwtDi+mrT+r/ygsofokyCiZqd+a3ih8MICpBbK6qJCtfy7v\nM2YuIx2IKBysDKgzoFs0v69ML1wkkq2jrzKve6t+MAwrF60VLrgsgDDNKhGrCqnNq68mbiuZLUei\nSqnSJbqwHadCsRowVjKHMx+uTRugKy2t1C58MCMxgy6TKn4rhim7nP2m1iwYKMawaDJ3MOyoRaoT\nr8o0tqwyrgEsRCwVsEaonaVssDKsLyh9Ipgo2aloMZIjOLADMKuq9Kb+siAp8q0nojokE7B5LB2w\nkTMkok6rJCsxr3ErgySHNO2w/SvkrSonDixyreKtlrCTKMIw57FdMcEwJi48J9kpEpnYsNqm5y1V\nJUYxEqwwNJ6dH7X+KPwrlK1DpuqwQLMKrhizV7EGsY8i76g3LMsv4DJUsTAt7C5brqsxwKzDJPcs\nmDLRsXgxfK3RrPEvUCY+sQKyzy2jLJYgdCy0JY8v8q2bKrIiVyXpMJ4eDbEUq2EotqxXqVu05S2U\nqp8uf7LJqmawNy/MrFOh4CdRLGamuzHbrPovVCw6sKCs06/ipRkxaa4IpYMuvyxMMNcueShspWcs\nDarPLiSu0zDlL/kqtqCFLIsxl5lmLlGyEi7lMOcvPa4krlwp/yiyCKSkOjEQrtcykioVLyovs6/J\nsGgp/xZ1rP8wES0dKt+yYzP9I/Mh8CygL+wdzSW7Ll8pJakpLWEq8qy5MESZXq/QmiKnrjPGMJOq\nfDIOrSYlijOpqHWwbSuUsNWrs6epsLuvwCE7soKbu6RUMMSuj6+HNKioZi50r7IpyaMmLxUqOKl0\nrQ8xUCMjKKAmxinRJoCmxqcuLIaopp5nJFMl+6t2MbkkJSFWrREtMKFfHS6tpC85Kr+pdDDBMF8y\nX64Kr/qsoK0bsaKxkK/YLBOvi6R7Lp8u8KjjLo+gkiRYrU8wzDJ2Ldch8jAoL2Qi3ahMsFUsgiRa\nMG2l0S/nnHYyBirkqtSfjqJ3KQqnDDRnLlUdIbIpMEolTS6qr0gstKKXHgwuRq5aHFSrwC3oMfwo\n3C3UKu4ouzD3s9asSTJ5rXmr/60wrB8tDJA4quopzClIKbk08S2XIeivTSxLrpIq2S39KoIjSjAS\nrlkp2jERodgsRS6fKAYuZ60wrpIr3aIrpQCveR7Jpi+v+jQvrJwxIqjgsbIlKSYoMH8rRy8lLOYu\nMqtiLYcqQa1lLgiu2izGpEuoRzOLKyuuHa6qLI8wKKo3JbwsBClLr5Em7iLgMPWtYjDmMUivYanR\nqiuwf6XepteofCb1qP4uaCvwLAoq0imYLDYtgKkuL9UtWLOKHq2o35Z+NLKdf60ordstOanspU4z\nWKXbJBgh1SKcMQSwNi4LsM0qTiazsfQyOSNOnpmiEyweKFYvWTDjKXSo6S/0KYauwTNTMVW1M7Jz\nL3kwtKojKuiwxa5wqkKx26mypp2l1zAfHSci/i1xq6gtBSZFNPkxMzTSKMgxIq2Fs2GoyBiOIok0\nYbC9qy4s0ypSLiKssDJRsfwoZjDkLo4wF6ednIYwUbASJcYbgSm9r2ClQizrMies8KyBGgSq7S9q\nqwkvOys0qPuufbAqrIcbpS81sQUf+qeOq7muziMSKI+rTyP9rMAlc7ASpUMtmqTirXMpxDLEMPgu\n5SgwMD6wnynTMKIjSK+TsR8x6q4dn0Yl7jA9oX2vHiSUpVut4C9ALuauMTEVro2qqaiMKLkj46G2\nLdqtdBfGrjIv4SruLKMobiS0LZCZBa2Mr1mpgiIJLYysqi0ztNIfU7HVLlSsp6HCLB2mgCsPp3Gn\nA679KsYlzjAWJK0xHCq0rmGhuyWbrYAo0yoLLCgylac4rYapyLDqrRWL5jLOrnutHzE7JGOt/6iG\nKtImAiM9KLowNC9tMGOtdxQlsvovHrHqspgqeSpaK7KalaQ8sRUsFqc8qhYt3S2LHGYsOChJqDiv\nySwHnMIscqhurukrzJ/HqVgmjqiequmxnq0sqAMtaiZXLHcn86oLoB6oeag4JPwvpqZ1KxWoVK1L\nqCikNLAhLqErOS7VLhGoHip2LJquI7OhqxEsCqmCMEktGiYSM/+kq6SMrGmtxaIBKpktxDMKMScv\ntC+7MyysHbFzM38amjHZJNCsqSzwIYuoty4Fsi2oOTGxrFQv0BghLxqpEKuBsbms/a+rnoKuQy4x\nsD+pXbIJLKKsAqwrrvMcRaxDrjmpijFpL26tZym8qD8pcC+xMAgm/axisAkywJ64KoKmhrB4LYKm\nxh+wMHKs1qsYGPCtRKymHuwkeLDcrDImmy/BJOyrqiq3KW6r76qCpSCoZ6r3JkMwMTVSKu4w+S6m\npt4o1qcfJyUsHCw2siEwhbBQsKUmHah7I20rwC/SMkqtNrBRLMOwS6aRKwIwQzCZrOMqOKnyplce\nGCmIpQGkirN5MKivRy+tIZAs0y48q6kgfavXLlIqQq0trjykGq1vph+t1qUwsUGwhzIsM8+pGSaN\nMJ0n5xqDLZah866nKk2qgbB6rxIw0jLArCKsIjD8rXgpkqYkLFAh4KQLqJ8uDCnoqQuw7ptKsCCn\nRi3CrbiiHbAVqrEsIy2/tPgyiq4pMGio1S8/MiGmdK1Ws8cuBSn2q9ssVqzfpXcuFCsnr86tCbLY\nJWUpnqFyqEwt2qKwp1esISg2LZytJaGOqbCziq+RMMso7ip/rnmguilXsGiwrS10pesqCpLikKug\nwaDWLMusDayWsS+wNjBSqcqiKDAPsh6z9SIUKBgs7igNMFESS6z6qRCtm6kfoWsvkyxILcmoDqwX\nrJqwZaiDsmOvha0joTsoEC5TMPSp0qRRK92vJCTVpowqUCxyM30rEy5IqZ2y5andLFqgKKhOM6ic\n3zCisDWjZqziKVqu4yYCp5wupKkeNIIwkqscqCWxIR0KtMuvuKbWGLUtPqjIL0czKitYLGwpHKi8\nsSypwq82sEoY0KxyrR8csix5KYIt0TKhp7arzC7XqpyrBCWxow6wl65drVM066rFHU8vGKKEJL2o\nHSt3sFqx7aICJUMnwayoKW2opDMdJ2QqQ7CALcaiDC0OpRqXZ6c+sEEv0KXSIGqoTC3fqEMkiih5\nLAcwMSTesMGsEzVoMBanFq2/rhwxmyUKqzW0KqpEnHktPyYCr90znK6HMHyea65XsLUkyagJpzip\nCjEbqKggkSwwqLKtr67RsmggmaoHqkMozCzHsicw8SvOL90Z+bDPMGuxC50NNBSuQjQeot2wDTRM\nLlckRaqwJB+uVpGKMP0sjS34KRmx4ah+MLyxjCAeL+evqDDttE6vK7BpLfgucDUqLXswdilVqjSw\n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\nWi66npSdwjJhtE8ydql5rBimmikApL8tuiE1NDYsai9qMe2qPDDdMRMciKx6Kleu0Zqcp9Yu4qBR\nrHutNawUMamyhavcJLWy+qfAqwcwGCCDrGMlCS9jLwemZSZEse4wCbAeMZCwqqBJK8Osd6jBoh+l\nlLC0pP6tIzMBrRAv8Ko+L0goyyZgpACsuKYzsvKsu6XKp8StaavYpZ+khijfsc4wraelLeymz5mf\nJUAqbq8DHj4nayQWLSytWLCJMYkyzSoNsc0oDKhprPokOKV8otkq6jGGLFkvY5ierGMy7qYhqB0v\n56ryp/Qh1CjgKUSYRx+hpCwz2DCDrBilZydtryIrf6WhocIoOiCsrmyo9S1grpCw/6t0rmwiFaYD\npg0wmy6EqJinPjPlIP4uz7DWKOukfSZ0MEc0jy6/snuxqzMkMKqs6LDHpXsw96psrzgu1zBOLPYp\ndq3yqq2otbBqLT6vHq0kL6ehurBpsK2wObH0HHYstSo+NB0wLa5aMN4hIqVGqX+qhSaJLQyslzCD\nnlIpKSpsrUAsIKnRtAahuKXqLqyzoazysKCOKq/iLVmtVycELiix/i2YKLkwparWJTMwoi3SIuwv\nPTI6raqgni4/q/Upcp92MPep/awNMLghw60QpKKxX6k4sMIy7aeFHzIqhC07KMGqxjFVofkws6p8\nndUkcSyrMJSvyShZIi4uHiWvtGwkV6weKLeqHK2dLNGw5i8yLRsfMB3ekFewozBaEGQstjAFKY0m\nFivrsjKwnqUsLxSgDSrbMqU0Ky3dqhwyWaqrrmEfXy9MMJmzc5+bqE0q6SxbLiaZiqv3sT2p7bI9\nMm4tvjDuMW+syy59r76wzy7ZLFcqoC4ms3Q0WCZRLuuoXDMHrxAwWqx7MC8uxS9GmXymvjFlr6mq\nKi4pprck7q/jnxcveC+vrM8t9qrsLbKxB6y3HlQrTy9JqN6pviwLK1Qt7i9iNJ8ox7F6JPkkwzWf\np7Eo5ijtLyMsj6pZsnEwYDBqplQsii7zLqaw5S09Jseh4yjRsL0orKU6qqOu+isasJgtOSBqqZSy\nBTINMV0nWbAVsCgmLKbcLn+mXypFqCskiy2TNNohbSW/Mr6z6aqgoyC0yCwXsICxHihLKTOuPrVk\nMY6sz61Er2wx062ipzonDiyPLVw0pi2mrW+1XjAbMOUwuCo5MF+oO7J6nS0xECaOLp2pMIsEqngu\nZhwksiKoOrAbLFuxHCwNpR+v9hQ3q5Yn/SbTr0AulrIfMX0ndCyELHCoVaUyr0Gr86l1IRqw/6IY\nqzsoRrGBq1gtgTHeNV+x+6nUKvovQ6fyLdix0qtIIzspArE1pxaqn6S9rZkfrCV3LoCxcDIrLzst\nqKmVpWszA6gIJqAn5ihTL/+s8ShCKO2prDRTLnAyrzBoLk8graisL9IqVC+qrDgq7KbpLsssUzEK\nLIUyQaYisKUt06h0snIypKwvs8siZ647JkOsWStdLscvDK5krnOoSy6gsJwtgLEvsnewIbECLumw\njjHIMF4sNhsxqciw1aKNLkwcbi4ZscQxDi09pQidIipysJ6tWCqfMEKmIKyorgMyATKfqjSodq+4\nrswtVi8VMJktACs/sKipqqpNKaisl6xfnYss5afLL2ylEai1rXIv0a4crb2vly2zJlAs7ql5Lzqs\nN6SCKwqkHqUmqmCrtrDlqTMtda7lpKkyjLBYru2oGikYLeKx4DKWqREv4y6PrQ2xMRoKooEz8TA4\ntActiqo2IWykLafkrJ6vK6veqiEyYyhdnJUnOq0erMEnPSeTsKqxIbJprCYiVq6LrpIwuC9sKDgr\n+CKaLT4srKzFtcAu6ClvKwywfSAYrNWx5a2usXowRCcVL4Ww8iMUp8WxRCzeJfusyzGCNAcqVqtk\nJ+Mmhyh8Mt4g4Sy3r8WplalQMqehPC5pI4mvTidlnoktaa7Esb4rzrAHrjqwPSv6LIswXzExLOAt\nECX9Fo0x06duM3mtCyW1K7mUbiIJssEiHjGUKfIqA6xDMS6nrCtPsn+kGqtDqxspeCihKKYwzrB+\nKnotxLBAot2peLE+rnIqzS7wrs+vUJwFqHok6Sr0LjcshTDGqGGpibBdrEyzDqq2g9sr+KxVqK2x\nZigzNpo0nDG+sNMtO654rEkvirGvMReheauoKHgxfiymLPyxia3FLJCshCTmrXMvd6+sJBmvESwy\nJiQnorT5sAGs5a5fsRGnz69FpvuokDDFqpkw3I/rruazADF+JACqkaQOoGEwuDTln6KtJDGkL1kk\nqSvDMeAo7TNxsEchjaQEssYuYy5GKUek2inNKSetFiYQKgokKywiqegsErJ/LugkSTHhpukrHKu/\nMawpMxy2KQyvSyhpLc8isKpzMnquxa2iq+Eug6OfJWaoUy0tsE2mzTUmrRGwSqxTHieS7auqqTKw\nX510IHet7iRhKxodIKCZsNieG7BhM5ot17RWrYirrCuHqlIfmqkLID0w8CB5JHYsgjLGMXqei61A\nL5ypby6NIqotOzWwqgouGS2oMMykVqp/MLKqYazcJlQp/q0rJRktwi/hLMIuuKpysauW5KagKUso\noLATsMgt4SyxL+KnI6mGGz2vdp9EpA60Zi1VtB6hQaxIMlswaKwJKjoiRC1CMVqj7Sz3Jecq9TAe\npa8sux+sr8yjvy9yK6clOC0hsMcoUJ1qqWQxLa5FLXwqJShQssirEbDsqAEshCNoJ2ykua4=\n"], "beta/b": [[6], "MTbKNX6v6Sm3rJqx\n"]}


def LIN(x, x1, y1, x2, y2):
    return y1 + (y2-y1)*(x-x1)/(x2-x1)


MAX_THRUST = 200
NBPOD = 4
TIMEOUT = 100


class Model:
    def __init__(self, weights):
        self.weights = weights

    @classmethod
    def from_data(cls, model_data):
        weights = {}
        for var, (shape, data) in model_data.items():
            weights[var] = np.reshape(
                np.fromstring(base64.decodebytes(data.encode()), dtype=np.float16).astype(np.float32), shape
            )
        return cls(weights=weights)

    def predict(self, state):
        layer = np.array([state], dtype=np.float32)
        layer = np.tanh(np.matmul(layer, self.weights['dense0/W']) + self.weights['dense0/b'])
        layer = np.tanh(np.matmul(layer, self.weights['dense1/W']) + self.weights['dense1/b'])

        log_eps = math.log(1e-6)
        alpha = np.matmul(layer, self.weights['alpha/W']) + self.weights['alpha/b']
        alpha = np.log(np.exp(np.clip(alpha, log_eps, -log_eps)) + 1.0) + 1.0

        beta = np.matmul(layer, self.weights['beta/W']) + self.weights['beta/b']
        beta = np.log(np.exp(np.clip(beta, log_eps, -log_eps)) + 1.0) + 1.0

        alpha_beta = np.maximum(alpha + beta, 1e-6)
        definite = beta / alpha_beta

        return definite[0]

    def compute_action(self, game_state):
        action = self.predict(game_state.extract_state())
        return Action(game_state, action)


class Action:
    def __init__(self, game_state, action):
        self.game_state = game_state
        self.action = list(action)

    def output(self):
        for i in range(2):
            self.game_state.pods[i].output(self.action[3*i:3*(i+1)])


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Pod:
    def __init__(self):
        self.x = None
        self.y = None
        self.vx = None
        self.vy = None
        self.angle = None
        self.next_check_point_id = None
        self.boost_available = True
        self.shield = 0
        self.lap = 0
        self.timeout = TIMEOUT

    def read_turn(self):
        x, y, vx, vy, angle, next_check_point_id = map(int, input().split())

        if (self.next_check_point_id != next_check_point_id):
            self.timeout = TIMEOUT
            if (next_check_point_id == 1):
                self.lap += 1

        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.angle = angle
        self.next_check_point_id = next_check_point_id

        if self.shield > 0:
            self.shield -= 1

    def next_checkpoint(self, game_state, number_next):
        target_cpid = (self.next_check_point_id + number_next) % len(game_state.checkpoints)
        return game_state.checkpoints[target_cpid]

    def nb_checked(self, game_state):
        last_cp = self.next_check_point_id - 1
        if last_cp == -1:
            last_cp = len(game_state.checkpoints) - 1
        return self.lap * len(game_state.checkpoints) + last_cp

    def get_new_angle(self, gene):
        res = self.angle
        res += LIN(gene, 0.0, -18.0, 1.0, 18.0)

        if res >= 360.0:
            res -= 360.0
        elif res < 0.0:
            res += 360.0

        return res

    def get_new_power(self, gene):
        return LIN(gene, 0.0, 0, 1.0, MAX_THRUST)

    def output(self, move):
        a = self.get_new_angle(move[0]) * math.pi / 180.0
        px = self.x + math.cos(a) * 1000000.0
        py = self.y + math.sin(a) * 1000000.0
        power = self.get_new_power(move[1])
        if move[2] < 0.05 and self.boost_available:
            self.boost_available = False
            print('{:.0f} {:.0f} BOOST'.format(px, py))
        elif move[2] > 0.95:
            self.shield = 4
            print('{:.0f} {:.0f} SHIELD'.format(px, py))
        else:
            print('{:.0f} {:.0f} {:.0f}'.format(px, py, power))
        self.timeout -= 1


class GameState:
    def __init__(self, laps, checkpoints):
        self.laps = laps
        self.checkpoints = checkpoints
        self.pods = [Pod() for _ in range (4)]

    @classmethod
    def read_initial(cls):
        laps = int(input())
        checkpoint_count = int(input())
        checkpoints = [Point(*[int(j) for j in input().split()]) for i in range(checkpoint_count)]
        return cls(laps, checkpoints)

    def read_turn(self):
        for pod in self.pods:
            pod.read_turn()

    def extract_state(self):
        features = [len(self.checkpoints) * self.laps]
        for pod in self.pods:
            features += [
                pod.x / 1000,
                pod.y / 1000,
                pod.vx / 1000,
                pod.vy / 1000,
                pod.angle / 360,
                float(pod.boost_available),
                pod.timeout / TIMEOUT,
                pod.shield / 4,
                pod.nb_checked(self),
            ]
            for i in range(3):
                cp = pod.next_checkpoint(self, i)
                features += [
                    cp.x / 1000,
                    cp.y / 1000,
                ]
        for i in range(6):
            cp = self.checkpoints[i] if i < len(self.checkpoints) else Point(0.0, 0.0)
            features += [
                cp.x / 1000,
                cp.y / 1000,
            ]
        return features


def main():
    t0 = time.time()
    ai = Model.from_data(MODEL_DATA)
    print('Took {:.2f}ms to load the model'.format((time.time() - t0) * 1000), file=sys.stderr)

    game_state = GameState.read_initial()
    while True:
        game_state.read_turn()
        t0 = time.time()
        action = ai.compute_action(game_state)
        print('Took {:.2f}ms to predict the action'.format((time.time() - t0) * 1000), file=sys.stderr)
        action.output()


if __name__ == "__main__":
    main()
