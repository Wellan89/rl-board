import base64
import math
import sys
import time

import numpy as np


MODEL_DATA = {"dense0/W": [[73, 64], "6SOFq7qtjKxFK0syLhOCoukpaK2FsmSV16t6MJuqoS64KSekHK6wKKujrSnVMgiwSiULLfqqGaCj\nrx2wha/GqE0uZbMxq+KvcqhLtN0v+qJzMGQvXCjvqEewkzAIsBMyEZq+LAKym7TSJj4rSCW7pdSq\nOCKJrkYrZC2mqMgpPy6bOKS54TU0Lf0i3DS3sGS3w7gjKuKy6be8tm2qkbb7r5YyVTattwGizynZ\ntsayAqy4sQ80gbcfNzU2ah2ALrScYizYOHauMjnItKI1ty4bLykvWLXJsck0KrUvsm0x1qVzsbw0\nGa2krVIvRrZTqd8rTjajt22umDFyKBon9TDksz6w0bV2roi4iCynNSQ1BbnXsxu2qDRaqD+4YjdY\nrE+1wTG2L0m18Sj5tu4tsbW5N0MtkDfPrJ20zDS/rL8p8SZGNCMqXbbvLW6tkbPgMs+zzjKvNIa0\n4DOitRy4QbGEpJ052SxesZ4xX7f/NCU47zd7MZg4BR6urDguG615Lcm3a6RHK7y0PzEoshkoW7ET\nMwgu3ytxOWouSTxTrzkxnTa0scGcCTK3His8/afGMKExXLApqBKwVaWxqdguGzOEMoszPa4Psni0\nrbBcqpgp0SjNpd81pKhZpYY1zDGApPCwJLIbKzo0uD5xMbMs8iyNsucznzBjKYsgCa5Etb+oSBjC\nMzGtxjxxNx+y5ao+sLiwwK9tsPmlt7CQtBc0GLWArbcyXy42r4YdrziNJ4IseiUKPo2v2bTlsEqk\nqLELpBkuuSwgMkYoXDeopj4wpbFGrBqzj6tHscqoda6HqWsuRBXArLMv4qofqgEyCrLnJrshorHE\nrIkpfCu5sJIl6S9jqAu0ZC71uDK3vKa7MrMkm7ILJtQwdiLutXm9lqU3NdAwr6/goIMrwKPtvi2p\nqa8SMiS9Hqn4LUamyau7pEOtcDGEM2ouwC2xJxQxUbPcMLeuO6+eLGwXzS6orMEetrDCrdEwaqWO\npmS5ezIKsowlZygorgqzBCRSs/WwtjM+q1iwlzCCnbQtmauEsJi0Ba79rh8kUqhTqUwxlSsUqjmw\nA6lGM9OxhrMAM1muMawVLvMpIa4vrDOqyCgCJoAw2CksJayuxTCdMActWydRsDkpx6wvr8KvZKkF\nMc4XfCt6raWtSyHcLK2vkDFELC6upqLMrDwxMDCOrcCxxymyMFymIbIkLxIw/rVirH+vlTDUL02d\nKyjfK4Qt16iBswqeXa/iqzqoDCzbKD4xHTYjKYIfejA0smsrXa3uqUGmsCnWrawqSidflFKulTA1\nsIwTaLIApNSoBqE6ri+kUa3lsBAsyzAfsOcnnzGjpYykXyoPr8cpmqNxKqYhoy3FrYymGS2JqhQw\nei03rVowkifMmEguOyptsU6q7rArL7Yoj6sAj9ExPq5Ms88suKnUrBmpwCvTrHmcyyBWpBAx4y2V\nJa4oBS7YrjGzZzGaJ/Cpv637LeQssTGUMWgxGi2HqOQTZLDBMxWxp6TZLUcotivYLtyuRyzRIjux\nGLF/rvotty3QIs6puCzHpQIj0q+SrQUx5KWjprOs7Kz/LxMwRKplqvkxoS2XLBIyqazwpJkrzi13\nlQet6SLZMEso2C1TrEEwR6sBsBC1nC6QtM0ldjF7LR6nGKpGMDat1SxsI7wxuyShJ4cu37KjNH8u\nUyqir1Ak2SBwLbutcysdnQcsBaciMBuuViwztLU41rQWMkW016IdtEs3xjdQL7s06jTINXey9DCT\nMKWnbq3XNdkoyKhTtM43/C8DNJ4nljOaLx41t7byqLk36LMJs6Q0F7hntQQpzbUypwY1TzBYLg00\naTdss5K19i2Dpz03prblL0EllTWKLD+xT6rkp+i0uTEvtUusCTIOMiSxmzQSLaswOSdFtAOqHjOk\nslO0ubUwsJM2CLejqEen0bLRtAskqiqiNfCsHSG6rhuqRzCUr5uyQ6Jzr5Cztg8gqB0yhbJ6Lkor\njq7RrjmwP7ARL0I0kTIJsVQy7ChKJ5yy7iTmMXizqzExso0u4bTDtbkwti5SrZUwaacZMNOy07V4\nJSMiCaYwJVy1/rXkKgU0razIrTQyACcQLBg1uKdssOmq461KLwubWjCeKsqwaiueMiqnYLVQroes\nYSkGJRc1m65kstgrmCxbJwGrzySxtKM0uTNjNMauRTIYMIQ1Uy5ztSizFRS3sQWvxKgItbaugyzl\nsJAsvjQfNDExWzGvsqUy+KPdqlip+DGnKb2wlbDPrEO0uDT0JWwuzyasJh62gDBlo58xfCHOMm0r\nWa2KJ4Ax+DdUoxO01bQxrrwt06ppsWixIi3Zsz2vlzXXsnu007CirzwwJSPtLGszPi/lHGYxMbAH\nsN80gq0ZL7iyT7TstN2hDLicoksxS6sjMmE2CKknKUGqH7NdKNwxbC2umcgTZjJNsU+k4SNQrFQ0\n7CKRs2OoazZmotUyoptzmpymmDA/KUksirV2KR+337RKsP+0/bCjHF4097N5L1qxVJSYsUKubjd1\nrbewwSysMtwt+7KzL6wuPTMwoRIypjDLMtc21q2FsTsqzLQWNE+zeSvCLn0oW62zOM206DS4MeYw\nC7XhLdyyh5sEG2sxbTIyMImwVJ36JZumhS8GNJSr3SNBLoS0+DQrJoyqRbEaMr4vUi3WDzitFjJ2\nNbouqihoNw6lebUxjFApm6oBuJ8vhqusrLSg2CqyrpAzJikTNXkmmi9StIsvUysSrok08K5WLCUf\nexHdtE2wpi7FMKecDK7xG7svyRRdlmCx9bSEq502wK5mM84hqJqOMnsvp61qFMQphrG9n2wuXDGn\nrpS2k7Q+sPGzrSFML9+q5TcKtasj/614rDirfrQUNMe02CByrRgzMzV0mOKtWiyXMJisBLVWNU8y\nbLJsqW20CapBrmal2B0bkYEx2Di7MVy2rCp9JikpsyqyMb6ZajMcp6YtCrZqNh0xdSgUFZuy0Kbv\ntdaUArWFtNGxGTP4snOpkSm0tVCwG6Rwnvi0K6z/MdqshCa0uDu2c6Z7tOuzRbAcp8SlfDZ0NX2T\n7DNQJYCyhi+7Jz+eArUJN4s2SrXJqQK2tKvGL/eopqV+LeYlTadyLdMsmiceLzep3jDXLe4wkaW0\nIaYn0Ci4qlAtfyXSqJ4rxKANqFu0VqQTrI0xshvfKPCiH5jzMCMwnjF3MEGoQafOqhCt6i3wMJa1\nfqo5oZ2uzSbmrk8rcav4MKqxnyorJcosYKrWK8wuLKwQqGKyj6hgLJMm3aykIT6xCK+yKGup3qgc\nL1KmJ6LSrAknFq+rKYUoMzEhMxizfSeWIVmppKzorhaxsSthsJCzWZiDq2YxrKrfrSWsHbH7I8Eu\noqzzMScrqLBMHgIs/6jCpFYss7ElrAixk7DaH34sYqWrrCIhF7CAK08xdjFnJtYp2i0QNGYwfaQs\nqQAtt6RzqJMezDFHLNikYC4lp7Et1SjqsMKl+TCJrfms8ycBLEqe4C6mrierMK9kJYstsC4PrJOs\n4q0xMUQve7AAq3krWy+YI6s0Qqx+rgGymbECqDszoLGyJeeurKmIKMot7K6HKeivZzDvr8CwJa8z\nMDYtnTDqMyKsaCE6K2Kl0a4VqQ+o46l1sAog7hyUM+aoyaW0pGok86ODJHGrVa7mMJGsuKpBqKqv\n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\n/6xuq5qs4rGgnFugwZ5dH4ewA648LHCsdzDaoRGxry2+MQMf+66xJn2zHi7CLUatsKq8nkgs36Ze\nq1yiazF+LIykuaxILKEllhwHqEMonaDorzKihS+8sFYrzKnRLzIvtiI4qOojfy20qMapyy0mKfMt\niahXJvMzXrIitEg0WrBpKkUvNinoLd6gQSyUJa+vTxNgKkarVauQMFKvwSaPLE2iU6ioJbKviSso\np0AsY6pWL2Siw63sL4Yn/CFcqeKwjyxXppYwoi9cKNExPTJ+MForNa01Knio7C5utdwtSC1CLH2w\nvLFNpE8tuKaWIeSrpayvJ8IqoyyTK2aeUiWvMbItrRt6Hout/6kGLByu3oQOqi6sdzShrLIsu6Dy\nMOww5izULw2v/K0tMiK1kzKfp7sxZ7AasHMoPydKrhcvbCjHobis368Rl0woBC8sEq2yCa8Ksgcs\n2DF+Mjsm4bJirH0mRLE3rcAy/KdOrp0kjC8ILqM0QzFIlDKsPJ73q46fc6yvMRqtTK4DLBM1/qPi\nLp6vvDF2r8AurK98rYesWa4RMA+zWiYHJiQpV6v4KlWrJC6po2ojZrCvpqyvM6kaI5yoGjKVKeoo\nzSEilykoniyhqBeqHzH4MGYwbSbgsO2yXRrvriwbdyotNHUbEy5zJ6EzLSE6qbmqiKmMruUonKOS\nL8ytuSzGJ1epSzAkr/cpL6VgLs+hS6p+K70zzK4ALI0sNbFxIEUv7CHbquCskCY3LEGsjTUNKBYg\njbCnp9QyrCwop5uv6y24lnwpl60pJyetISzMMQE1vLIvI6gmrCTkK6cxayyMrk4pYbDftM6vxTAh\ntvOuJihKtF+xFqRLqzWi4zO9rcidYbFkJg4q2bAOo/myCygwlM2kMKyLMkqiOCFHsrAsgqdhKKOt\n4652LREunKxIMViiwK8kqMYkoKJcsTAmBB5jDAAuxaz/FtmtNTLiqEoskK+oLjqtZiqlqB0v1Kod\ns9IuAC/3p2geZSwaqrynHi66JAGuayl5qMqsd6zvrY0vTS8zrskvSi5OJqQxbaLNLJcsqLH2LGMr\n0SuLJZGoirBTLCQbxqGwr4epWZu2MDavYKkYLYqiTC5bqqEwZRurLS8lOhQSJbaqdSMNKNco07JY\nqjinxzTOLawlqK4Nrm6whSlArYWwrKTjqekbLLDvrCmv6amBsIWukyTCLWSZGC2lsEMqEB5WrKmx\nSyUbqFksOLJ+ouQnSDCBKB6qmi4lK0sjMyZPruuqQKhFKVUvG6q3Lkew9i61MQWsvC/SpqSr2C8J\nqGotc6g1qi2t9TKCrFkwXao1rXgueyb/KPox3TDuME+rcyvNqqus/6r5Ircpvqo9rpojEa8cIdQq\ntC0XMQQu/aktJh6quy6+MFkw7LLDKZgr2bF1sgilBR1ALierMSsaMySwp6xfraIomy+gLyqyNKnH\nMvOsAi+XpSmtOayfoCuoay+vNOEtoSbWrF+uZzIBK+2vWC30M1ylySeDpFumM7DmMB4hgyy5Itwt\nsa4mLCipS6x7MhakQa1Hr6UtUCjWMokndiW2MQs3haQMsLenvyDhKFstm6idsVuvNq8iqNGneCwU\nrMQwfzG6K64yby7IsAcgqi0Qq3MuVSidnoSupyxpM2CvGS7wLcYocbJNrFuuZamSLxSxRqWlqUmo\n7a6zLRSqkK0lKOyxp6igKewuoC0hpWqxGCwzKQAsQSOtK1CyazDPJeeuUiqoMWwrzS/1pBSzA6ov\nmmOso7BGLEQ0taJvLfIuSiacLIklGR49rpWy7K0+qL8sOy2nrY4qz6f8MXmvFrGyssGx0w+Kqamv\nBK8HKRowlDN7LH6skS3jKFEv2rBHMYarwypdsc6lkiSQLlGrOSxQKvKs2K2XrkuxxKmFr1utwjB5\nI6snxi7qMYIxsrMarrqleKlNJA4gr6waIk2s+zDCsKMjTy4jLhSveBtFIEi0DCubrT2tLaZ5or4s\nPCbXrM6t66yWnKGlkawwJaQuWbAzKHEv3x/Is2uxDis3snKpdZ88pNUwFrCvLDIcdq3ZMeYpA7JC\nnakueKuMLKutEyrHLSyjJbGOIPmleabpLNohMK/YqGGuSKnzqx4p/TCsKh4wzDDKML6sfi+TMB4q\nAiRzM1YyOJxLsCQqIiI1rAWmqrFxsZewAbEpqu4hJyIBK2izuB9uKWOzVy26qfOtMaWwsDou2KyF\nLHwyHjGarCktiK0wsCqqNCGSqyMqDTHDMqUpDw4ALhYy9S5kMsWnoSpzKk0x2KfJKaikEqnrLOEm\nMqaAIwct26/aJGWeIJt1L0MmyiA5rGOiqyoPLOKabLELq6mgGy1HLFgq6C8ZK2uq+KwKskSwniy3\nH8OoZjCZsB2qZixmqo6hp6jbqAwwZ7ADJPQpK66yrx+pyyo8LpYxNCwaNFwjyqQVIk2lEKSCnSYp\nryjMLrio4TAtsi8xR6xTscOddimjJ9IsG5yEMjQuPC+JMEyrXzC4MTMptar8Kf2vHSdUqsYsyKRv\nrQ6ttqpHL/GwO6tDKFyukKriq6Yv3yOrrJUa7iufL0iraBqqru4wmq8MLsiwxSTVLXCtJio5qh2m\n261bqb2uNjLpqDsvs57ILgkyeKqJrCijo6kdspitMab3p3ytwKsMoD+lqCrwsTQwTKC8LEeoJi1/\nKUQqm69TMAgjCaQcLV2u0bDWMRQyzywlsYkpcKQ3qlEkJCEXoa0qKjKDKHUv5Z3NrXYz76ZgqRst\nPqlJnk8mWSeVLFijzJDwqI8yCzCira6fuicnrZMtiCb3IcOkpS1gsDit5CF0r5SwsqhFrt8kD6wl\nKdsvJS08qJus5DKsqVsxw7BuKk+koyEXLzA0CC3ssjGx0DGUMXasnLB5nR4wS6shrpkuvjCNKVOk\nW62ToYWrY65qLf2uFLCOL7QmGa+HsTqtRbLjpyIw2yb1M/guiq1hMMYrU5TBsBOuIycwMLutkC+d\nGGAqIiparFwsdK1DtXIm2aVLLuKz8a0hscYeOq72La6tmythLRKxVC29JgkxC6pYoe8wPi4oJUcu\nnzFQraGkFC9Rq4ckl6hxMCGiWKweMbYhJq4eqOuxbqoIr88yDoHpos8d1S/8JqyqvzEOIgAx9aje\nl/QxvpX0L6CpZhoypXYuAS6vtIeXIaqNKiqtw6t+LKaxkDAxJKsYbS1ioHWwgDB/KBArni3TKE0i\nPimesuWxrxwEL/4XSij7MZQ0jysZqhgyxq59sHUePzGzMDmydZ8WoQQoQSwmKSQetqghr26q6LPv\nMcQs6TDSMdWnyS6Jq66wFzEjLLYo1jDYs2I0TiSSLfuozDMlr1kvxa31MB4ucS+/mVekjjHbr9uo\nIS6PqX4jOLDapWMuei5hrEkuaaw3L5CxzqrZLFosOi9EqH+pryzcKpArAzA4NO4jkbB5JAklUTUi\nohswUyoPL/kuwKuwslEwGTBBpmIqIy8DL1avJS1uLUcsOSaHqSwcbqafpnWx1SuIo3ovObLFq3ux\n7jE/MH0olrBOsFAod6OmLrarETN+q9ikUSwcNHomFykSMmiy+KibIF+wSy0ksBCxfioqKVuvTbZs\nMWOsCK6woWwxyq54rjorSinwL/Mz5jDWsAe2tiwuL0UwdCVOMRSo4rAFpBswgSdrLCgjHKWerdgo\nDygnshsdiLE0MEGyiy6HpU6wxxQ6reMlfSlDrgwur7L6LVYk/ylOLIOq66TTrtStdqKAIqevXyTb\nrAMoULEQqz8tbTFrNVexuqgNKtQwRKdWLuCy5KyzqY4qQrEnJNCs2KRepBCdgB7xLf6wczKUMPUq\nhJwqrOIxzSuEooAjDahbLw6tkiyfGLqvZjOCMFMy5y9NLvwgzqqUMKotAC8hromoOqoHLVgsUDAE\nLPIyZqwVrbct4KW1rYwyxqywslQlk660KGuvziuPLRItUKlkri2oYii1sE0xCrHNsq6s6rGNLlOu\nxTBEMHUoRiTyqLKtwqUqLNMjCyqWrzYxBS1kqigNBipQsG6t2ix3MNUeRqzSr2EyczFlq2ms163q\nructKS5ALwAqGyvqsBqqdqn1JAmqvqyfFEIs/6O3LyOipKbNrYAvYrACrXuvHSx6K1AsUqtpLdio\ncLAMLd2khCQQrZKrmLEdrPIqa6/zkLYyoa5arPiirql3JHSvNTLzrN4ogy6drS+wZywrJGkyEDNE\ntIQs5Kl+qFqmaSnXp2yvgKlXsLYxUxuvHDggJ60nq9WQmCrKsH6xxbAgrM4gFawTrnswxC7Bpmor\n8xgDLZUurayqtZgrPi8jLjiuR6S3pjqzRq9EsbwvNZjkLX6uZSQCJC+xHC9vLiSt+DJENCgpn6UJ\nJ+cmvyp/NFonJyrBrsypz6hKMtCnVy58qEGv9Sc/oKYw667XsSQs+LBTrhuwHSncLJwwbDE1LFUs\npiQRJrox36c6MiCw4yRxLFsoDSkKsr8lsjAgLMwlb6lPMZukfyqDs1yoMqsdqyAoFiy9KOowH6xu\nKw0rpbENJEqrSrPNsLQkui4mrYiuZCTPqagssComLvEr0TBeqcmoRK/cq3uzlByLoWUsWq0kqXux\nDClFNs80uzGwsK8uQa6FrDkwtLGdMdaeSa3xKMMwGitML/yx7a2OKSgTXTBAqboup65JphSwex8s\nIzEc3LRmsOurKq40rhMpbRmvqAWlVTClrVMvap/2rhe0XzEerGuqEh9soDwwtzSNoMmtbC82ML0l\nDyvxKKYotDOLsL8dK6Tusc8uby5GKcejGyyLKz+tRSZpKYcjDiwUqvksibIMLqApSTEKpEsqV5rE\nJ0EsnpgCLKGvryTeKf8h6apKMmutxa1kqbUxTbDVLbesHi+UsECrbzX7qhKwra2iJpAlfKxPoSiw\nEB3EnbaszCVzLI8pZpWSsKYgIrAQM5ctEbUzrXOrryogq6ggNqnfoxkwix+xJdYrdzKnMVOoda0w\nLW2muS6IIn4uNDXpI5Yvsi5WML4i26q9LuSnvquHJakod6xZJSYuqTD+LU+kvqsasQKd2KgUKLEp\noLBosE4ugiBlL46lHqlAHoGvdJ5jpGCwyS1MtJyhgBdPMjIwYKyeKRAiRy0uMSWk2yxLJhcs+TBY\npaYs1By0ryukYi+BK28hVC0fsMcoNhx9qQoyZSpuLRMqpCk8smas1q+OqCssuiJ7KG+kKK4=\n"], "alpha/W": [[64, 6], "+q9Qqxck7rGgrIurYhm7tvMsCaoGs7YpFi0cLgQ1abBrMBKkLTgZIE+t7bDrsBowZSscOgyqhK5t\ns0WteqTxsdsrfaAcsvm1HLQtNZyrLB0wNJ0ukrHaMvkwyTADrw00pbK1O9CvS6QZMrgzMjQpMMws\noaH+tY8wday1teYwVC66JdYdajKLMFCsuS8aM4yyB66rM0EyRyvPq70wLrMYsnQ0rrS6pnSuMKTM\nqkyw46N4L44sMbCYNJqmmac7o0qwOKfJsS0yw7SpsgEu47KgtG0wJqhBtbKwJ7DSNXyhUzQsNUis\nCq/NrNqfgbMpr1QxvbRSvDk2ErQlsMOwtbRMNYemaayip/guJ6waOOoVPrQktGaptLNqMGSuNK9/\nKLMm2yllMjSxxC7IJ7+kOqGRrXopAK1FMHooBiNFOJUkTC81o0mx0KPUM7Mo0rOEpwk1iDBmOfcu\nWrMzJNyvArP4MDE38rN+tRE3vbBXuZSqyS4VsPWpE7BFskCx2Sxnrb2zKDGBMGM0666Qs0I1hLII\nnrczSKgsNQup6jPRMWWlY69TsuC0diw4MLglXa5os2iwTTYLLR4tmbB1GAMxGKs3NPapBbMeoTyw\nnTPrqz0psKzkrDwoHDDcrDWkFTSWNeGwubO3qHE0Qy9wspSyfSUbt4qZerJeKYwzlyu2MDsx8SkS\nsb8yzagLOWejPaYItKSpXbEhNMUg3awfteO0pDX3KgMmIiVrqb41LLYHs0iwSzDsrwq0ULZMMtQo\n/y7/lUYwWDBNOFS2vyx9Mu8mjTjLN54oiq7sJWEwL7CftrYqhy89JkWlILGYMzGoxrDZsqMv8DF+\nMFWikis6s+auh7aytWswKjTpMEatsDShOUusOrIcMPYsLzhoOQavGLSOso2y6DXKOYC3Iy6mqxYx\nozSbOlAspjHBsS6uNzJFO+KU2S5pJS2t1q5jM6A1G7FoLLkzUiYOtuIpX7H0KUwxdCTONdYyUyx+\ns5KzUK1vsKKobah2LwIw+TACNZ8tBDOUp2yn\n"], "beta/b": [[6], "rTQpNVqwhSrBqnux\n"], "dense0/b": [[64], "hKR+JuqnVihuJBMhLxi9JQInXqRolNScISSOn/0jaicPoVeivB9OJCgqg5qoJESifaRXJPkiQYBR\nnaGlnqTzh9wjrqT2nPmjLSALGdShjqOgH/SZZJmjJPKd4KFtoJoeQaU2n9OkR5HwlswhhCEbpUel\n4BhXpGkf8xVOIDEgg5E=\n"], "beta/W": [[64, 6], "NjJOMWuppK8wMfEsQzQVOBy04ageLLCzlKwGNZ0tq7DfNBglB7qpsncxUqw8MYgiySuvtUozKiuR\nNLErBLQyK5azaK6fKMIuOw6HtUgtwbAyrpimuTgLrTcsLC0krd2u9DgUuaw4fjOLsGix1C7sr2Yx\n6jHMMlswWDW8NhKsOC2QqTiwsa1qopMzajITNFwqbiqDtqMqV7AwsT+1Iy/EMRyxv64esCmrTbFI\ntkqwzzEQKqqygjfjK/4yILHOGlowlzWfMWSrnrGiJK8vLTXVNaSzli9ProQvZrfktpaqZDH0qJUt\ndCw+NLquaCY2shgtIaqoOlG4uK/jsLSVzzqqsVUxn69erMMsxjaEs9KsnTRbrymqAjBglZ42uC/c\ntCowL7UVqGSs1KsGMhA44zTqNtoyLq3MsDuu3DPftQop5C/4Jt4uEyxqtNSxmDSlLAesV7Tft3cv\nxS35p6KkszQgM+Eriyf2qFKwR6pwNN21GTFeKAGsyrR0tMgqmywMMls0hjMxN3MufzFSKgwjtzYj\nNcktLLFNtCWyHLjhsX2ura2Dqo2zpzdyMMSudqigJcuzaCt+MZ2sxKcCuK+2wTZ/rsssEbOtEZQs\nvbLcKyG0chTPrvas7ifurrc0ebG/n28rIzkINI0xk6/IqTIw8iXoN+0s1zUpmP+vzjckMeAoKi87\nMByywLRwub4xn7Ebs26kpTPeqRmu2i20s46lP6hmkACqQSEooLCl/rBlIQ2pQKlTNA0xZ6y2tlOx\nn6evtessUbV0uC4l0697KqIwuLvptNutH6eBrle0ajPCNTI0zqtmGxIvvCx+rIsycTIxMCssxzSg\nND8w+7LqrHwylSmbNsK0e7FTNKiqVbiRueMwly+opgkw17dwuckzqS5xsK0v4LPwt1s0CDCRNWEp\nWK3xuGA0YrMItPcxx7DytYQ2ra00sHMuOzXKqS6tFaufsWuxj7JLN/UvErL1KTAvxS5kuO0y2bAG\nqzA3HimYsLCvjbHsKdmeZ7ClKcEuBq3VM3Qs\n"], "dense1/W": [[64, 64], "N60br9CuEBRusIUwAKzyqwQmVCrYJ7sroa1KKpOw2jDnqnQwkqJIKfeunaSUqlamK6ylLiszYLSO\nsdm1irF8nHOy0i0RrAStQCyvLDmkUKxJrRAsdCUyqnubFa27ofwlc7C7re0fFqg9qBymW7FrI1Cm\n5jQ5pGqNsa8iKWYvK65dpoWwCivdqSesQqpPMSaw37bcocKoYbSBMsKwD7AJqtUoYCnFLtwowrI1\nsV2u9rMNMUIuI6q7q6QxlDBFtI6uKCYJMIMwI6SroqMnti8ZKcqnH6yYJnMxBDRwM4ywMDGnLmSh\nGi0lqRsyVav4rKMo8C/rMQowB6NHJuEqUa7MKgUvczX2KK6tgLYSruayDq+LMmeu+zKTr6K0ejQd\npd0oWzPgMfSt/y5tOBotMKvBmgSyPC8srVmw57d8rGMzZy0vMY8vuZmrofioYqPhsVutdKxnNcGp\nvrX8puWZ1S7psF+3Q7eENW2y8SkZNLC2YbYsuNmtY7iBrmQ0PLC7MfGykR3ULHSt5q1xsjw037BA\nrf6uei+AJ6Cu0i2CKuEhZa7mrdswPigkLtQq2il2JuOsh6kvLJAsgrLdrPMvvqyqqysoRDBVMyMs\ndicTMAmtAzSHM62yC7QIFkExrKvkKfWqW7S8p8moGrD/MD2tpCVMMjgtmDLPMMqpoiAWNqSwn6RE\nLMgs+C7jLCQ1Aypgr0OdYKpSGtuqKDBrrPurpKn2MLyrfiPJtegszjGnLIWTmyrorR0xTrOdoY8s\nLjPMKBEr7CxPLFsw5ywnHbc1erB2qqcsBTC9MSQlXTBNqRGwIrBKsXivzzEaL8msFaGdLX4hm64k\nsvwnsS4uLPSsvqEgqVeskSh3LN8tJjWHrFOxxK3FsGMz9il+MVqtiZ4zIAQnqSnEMf6rjq7bL08u\ntiZNK/ktx7EGDEMt3S9pL20rECu0MKSz8bAAsJSxy6NlLskporAgLC6q1CzGrSItf6rLsBatpbAy\nr4Ixaqz/sB4vRjAFrWKtkLDyKAOvcjS6s4kx7zCsp/ytGzLEmnqxkSzHKd0NbSpSsnkcorAjNFQi\nczIANKow2KjvM6YtYC+ELJ2m3pNCLfutHSjPKlox2a7OpZ2sVioqsAYqOCbdMtqo7yw4LweiPLLh\nJ00xIK3ZqQErsCzssM+oH66fotYkdxyyrhOwO69imGQpGi7sKNgxp7BPNDWpCyxdtZEr0DGvrO0z\nlrSGs/OrxLB6MQIutxzoKIw0WzBksqcjm6s6sBWx8bQitCasNKMmNc6rniruJtGt+CoFKk6sqCxq\nq6Yvj6pBtZAvZbVjqo8qDCkdsz0wyTEMq1so+rI+KEOw26zxsM2s7i7Gr5gsMzIjr/IvzCofsycr\nN64FLlir0S5WnAWwsTAhtLWwg7QrsSWmNSBisIsuazINrJsuDKZ4r3iq+q9lMAwqAqzNLB0yfivH\nMeKyEbL7MJcxHxh5LVA1LykALpyy8zNLo9En+K4fMDMmP7ItLpgpQC94LYsf5i46sCUnczTArwiu\nwDDVMYMrFjHpMHOxaCvFIrgsjax4qMQoyqpjK2mtR7DilkUrTyKtrYip3zEgMDOxRCgorQUsri7b\nI0CpRDR3sVougjCrr2qz56yiL1EmGK0koQKj4LBYKNipJio0pIwwyyqLL2gxWrFuLrGtRizTqXYt\no6lUKwEvnCDSqAMukbA8oZ+sprDzNQerNjFmq26sVa1kM4mu5TCbLomvuLIvok2wZCtlqjUqSjAe\npXOh9yIGM3SpnavGqfSziilQMYSs+StYq5IuaK01sVUwxzGQJh4zebMeJlMpUzS8LPavIq9LLGYx\nBqwgMGGwfaVCLrAzfa1+sKSy8azTMaUzpqu/pDYsZq8sLVEhwS5tLS2lxzHNp3mnASxzMKqu/Ct/\nsS8uwCLCqRYbA6dBJHasuy+CKuIsAzBnr4+ygK95sEYr3J3gsRqroibJLcwz1y/oqnSusSj1qGQu\nBK95rPKuzy7trrOzGiqEM6cu8iOGLGMnpRuxr3OvWTK4MFKzbS1RsV6mOC3MsDoUe6h/LIOv+jE4\nNkA1GaZpt+gz7LTGtay6ZLB6M3SvV6PQMv2ugrT4LlsmoLR+Jdw77bjqtg21higqMVa11LKWtXud\nPTRSHb4pYimcLvOyLCzetuY1Dq5SLJ+0n7LGtuwigKOYMfyu4bWsMxwvRrW6Kzw2KbJttPS3DbYa\nMoCtiTQ0uG0hBSmsLZupECgmrHCh4p1sIQcwGDGPqZgr6a2cKYEeoSxOLmYonbKJrMQrhahNpOil\nACyZruqllSR4MM4sTCZTL+GwWK+Lr4asXaxqrksm8rAJNkkqRDHnKQOwByKVLHEqxChwLfquAi2c\nrUmq76c9IqUrzikhst6zPQfKpb8sIjSTpX8w7auFo+YzCyrAqLsrSq2fKfWfxCo+LbEuoqqgIeQp\ngjAWlRaq+SedsBWsZrF2LPWmjSmOro0h3Bpaq/8o6B8LJZSsLDCSsMAtBSVDLd0lFqxDK3gw7J2W\npSifga4sLTgxxyuXrU2uMqqqsIcvEKrPqwi1JC+apHeskK/rKX8xPyqNsCUtpSBqraEusrCls8au\n+q7srTQv7SNMqtksl6gsM1Ow6p/TpnkxxLFks2asRx/AsbGsXaoNKkQyE6tCJi2x27EQMrSqAKRS\nMzmq26oRqSQg2CiEHeipZqydIl8saC6/Ma6s863xKiIoCTEWMIOp8ihmLNct6a1IKnWt36sWLuOl\n2bAGo26vwq5uq2MskzPJMTsmxarxMDoqRCiTLzwq/DDjs8GYESzVIgYxwrAUImgt/q3vqMyvszIZ\nI8utOy8eMSEwNC+8rAotiKZNKLwzRLGMLNIonp43NJMwTanaqNKtQyqWnzczIybGqJWwraxVMyeu\n1JU2kK4wRjJBL0SrAK+FqFot7TNcNEOuuSsfqpYvQ67nMCGw/KlrL5cqIKlBMYUo7KaBJCgwlSk/\nKy4uSq9VKG+h9C1Ls3EwMjQerPq0TR1eqY+v2yi9rTosxbBPKcoljbPzL0azpTHmLWsufC3TqAqs\nAKsaKXkoBp+LMRaw6R1OsnSofqefKUyt5ysOp1GqdLD5rOKh9iIOFwe1N7B4sKwtNjNPNLwvkjAm\nqcsyVS1JKHaqEbFNrP4lCimGrl0suy/PM8qnramYogUsOTSTHMq0pzNCK6UukCWirJMuN6lypcQt\nfbBVqkQsUbGfMxe03DDDrUovViwrK4OnkZ+vmwAzTa08oCQsZa4cqRQviSYArv2pkyjPrf+qASzE\nLXwsCa56IR8lJy8mr64lS63+pfYiNDEkJQY1KyUOspgpUyxvsFErMTCOJlSx/C+NMB2syyUNMp2s\nli9DKc2vnypjLfOsfjO3LcWy86sQsi6f4imwLGGvrx5Tsp8zITByq2KmpaxJmFiwmCmFrgqnhLJ+\nsaW1VjWqr3wzjzQWN9o0C7VuIhMyQ7FNIYA0TazMses0GbMrt5o0iTStHPYt5rA9OYUwSzY5MFa4\nGi1CqiAicTXBLv0r8y1Vpc2r6jHGuIWqzzj7NViv8qBxNGc1YDU8tX+uByRSsxI5ijjxNao6zjgw\nLza2tDfVqlAwtCiXJWosDLMzKmuwhqM1pTWmmS8jqByyYKzgLBwXm6mpr+6sGrA0rpagy6g/rxMz\neKwwpLaxCK8drIM0sC6JLFcwsasIqeQp+DPLru2u/KZ/sBSpWixCJyUjqy32pCYsQys4KOEihi04\nq3ex6S7aMLIpfK99K/iojC23quCs+7BEqyozrjGTsI4qhSbIJ02Zvig1qysx5KThLkOpfSDWrZiu\nkiTYsegnnLCHrT2xdzERqAAw56+MsuerKDIes0SpeK5gKmayrxsvMUCwdyq8Lv8x3LMPLesxfTHV\nrBuwnrHALWmvYTA5rUMwcJz1IvasM6hKJy4wIDL3r+2jWjJlIS0o/y/xpvUoVi/CramZKTK7sRMw\nkS7aKmEpCKU3sB0qaKRprfol5yq5rl6unapTL6YQS7EVs32t2iRaNGqp9i7PKcGwtilXqSG0u6US\nrQIw76TNLzWwPijJq7Am8rEDL8CzPq14MWsrKCnzrz4tJDLIJiSw9Ka7L14urzJPKpCogzGqNZes\nbrp6tHkyNrDVKOyw+aqaNV+tELBNMcSyHTI2ND81gbOkKPE41CwWpy0xqqb0MAwpLrIvtWGmPTA6\nJtiz1zO2s9YsIbOylYmuKirXNhmuILFdtrMweLRmMTSxErV8ubE25KtZIss2Y7diuVq5QSerswIr\nljSoKSCuP6+WMzmujTBELTYyd68ZrhOqeCa/M5amJiPcKHCsFS1KLPooyCYzpa4vt5k8p3AgA67u\nLLYQtSbcpsYu3KufsM8uBS8rreqw3povMHyqwLCzsCQfXByXoVCwua9+M/SIJ7H2qG6ut6ciqmMb\nAatjLigw4zI6JGutCCy1IScpFy3Graevay4MqIOqkTDdrjI05ywSsb4rkBzkMBYrEa/onzuxk7MZ\nKMswv6Uzri2phrMGsMWtfy66lSKtHyWXIxesqrCqrrcsILCDrb8yAKgkLOmqh66uqh0po6uprDww\ntbOLM94p9qGZKHGxNa6mr4EslbJwKA4s1jFAIGYscicxMuqwZ66oLVotHqpNrf+tta1dLkYuZa4t\nqx4wtjI3HEMkRbC7LVKxYab8q8YmqDDQLlwrFrLZLWsq7i7LpVAr1K+NsfcmlSXALTUsYzV9rGor\nbiwyLrMm4SBNpl+nXiczrPwmgC/eMDsln6xOMWSqoi5frmOy6i76LrkwkCEFqUMcYbC7Lz6vriLj\nLiUqlyyMLlYMy6n5scOsxqYPqSss5yWMromugixgMl2v2qyTJJiyurHMMEAtraV9ruOt7iNKsXSm\nxC2PLJIura3ttMGs8a/iM6cxXaxQrP+tWaCUsx8uYTBDIWColDAzKLQvYS0wq82wfaELsTwtlan8\nrmsxgKhZLY8rnbPaqKcz+6bnIGyvn7BUMIewoLHYrGOx9qTKLR+umS0aqouu3S7nLf2tuqh8M98u\nujJ7sI0wxpHIp4StpS28skGs+6K8ryqr4CgCMaOy27HqNNWw9K7fsg20S633L4erxSmWLgQU8aUG\ntFExWC0Cq6yqbq3WtScqmil9pbyy8i3CqDitial3MKUwhCiyrrWeYLD7sLQyZLILsXUgUbATsBqt\nTjE2nzuxOrFzsa6zeTM0reQwjKd7K8mymiAfMIarAaxxHxmy5KuIozknm5d+sn+vWaWvJY6s96lV\nraGg9zAjr7QpOq9vrUap5ixJMOInlykWLT2uYrKPruukUSxMp1qULCZOqWUlXKxFrN4l7ig2qwww\nwSvVIuYo2TRIsN+oBK4VrviwViVEJYapBqRis5Wem60BMXQg2B/9sDS1giSaKAQtTzBQpf2oVawg\nLU6w/yjjMQStKSpzMd6xhLCPqngqEajQoYmw4a0mLCQvsChVsoEvqSmEpSewMqHYM2AgjRFjr6Av\nvDAoLkWsyrEtsMius66PsGgs4aPrLEmw0LAnKSimsjDMJRgtVjK+KzQmsa+MKlEtdy8qtB+iIy/p\nqyWwhSsjrFwva6ZTqjujF7J5rZEuSC6OKjQtoaz7pGozVy5csOwtDapQrBcxLqORKCmjz7E7r2Gq\nMS1jLsstPDC0LQArlig2L7Orfq19KA8xojPLrw0x+7Awslsnxag/Ko+w3K0usqCVlyZ/qaO05CoQ\nr6IlgKzrq/urZCT3MTyoqrJArU+wJa8yK5cwCy+GIQWo/68drnKvGqxyoviybC3jseao5awFFcos\nca+5qRcwjy4ssJerMaODpzgu3ywvqMCzCyB7LtQvBihQIVauOrKirvyvJjCAs1EowCB5mZmpvq2Y\nJuapvqiipYqu9zJLLZ8shKcerLki2bHnsvIqOrMzqS0jEa48JUGt/61crSSt8LGftG00nSYeq+Yx\nsyZ3M36rdyzTqbMUnLT1M/kziC3+MW4vZbGxL1Q1jKIAr/AqhqsFrBEx9zB+IGGs4aiSKuOn9KmM\nNA6097Szspk0OiUbtOSXjqxGr0uoZTC9M1ozQif7MQatnTBLMA21cywFMEmmTa9dK6y0MTG1rokw\nxp1MqxQsBigFrjEoHDIKsVExcDEWKe8w/TJZr8St5SNnM9Q1J6CnqGow2ixXrkswsSo3rr4ml6+o\nJLArKy60KTyhky2araawEjCYMlqo4SL/JtQfwyblLfCknS4eKqIsEa+MKNenvql9siYzeigcKHCg\n/COfql4zxJlKMGmkoCy2LMWsoy06LGKzsacbrIYykC44sLit8K98MmAw86itnNSuSK1eq4EszCgA\nMiyfUbCGqWwmhLE7pbKwK6F/LCuiSzDZF+AxSqy7ld8rqi5iIgCxFaqbLnWuWCwUIK2gxrGvKq+g\nHTHZLPis5bHmqMWglbUXJ60vYy4sqGkonx/2oHMYJrFOLdeptx0gJgMoUC3LKKuxYS3SpkAqlTCf\nqDWymK1IL0ixky7WsVIj6iY/KcslmzFLsRehKCrPL8KkYC2pK7aoKC4TMaEsPrOoq4Mm5q3crM2w\nf67NsYYoL505Lk6y0y8Lrposz6kCsNAtIKSNslsrrisSJu0xzKpApFQsFKzMMYyhoTAVq62vKSIj\nrNOnTLG5JqMpCapQq+Kpt6fVMMexT7AqtIquaZ3XqdKuUKVkrdCnbq4BKXWt6SvELf2llZ1jslyt\nz6ufrQOsLxcFtFEsHCmZp6AwcagVsUGo5zErLIwgeaxasNytlaUaKg+uJC8QFcyuyTDOL3Kvqq6C\nMP0JQJ/or5woGKCDLekwSDCbLLuoXisXrqYwLy+UFr0utSlkplUymCs0py6fqSsFqRms4K1xr9aL\nIrHoq4YuCixhMSgcp6BJMTGtIzJBniWwgKlhriMw7aW8Jt4uqi2RpqUwWzBsNhGzfy7dJ/cb2jMU\nro+xz67PqHmzpKeirIcq26skMmwvGCHSKuQsiCClLVixnjO6NNWoRrDNqEmjGygjrhOuvC/2JNut\niLRls8euujTqJygtmaSbqFiwzaSSrMCriypVL5ov5q7KqN+tkiubLkGufbAwIAauFyJvrb2ww6+U\npo4iQzDgLF0aDCmhKA6xajTsJZ2k+CkIpsqwIzRgo2On8bDWLACx+SKVtFiyai45qFAvkpM9rTsw\nfzSsJJOwiyQfrBSs7araslmn2K4+riiwaKczr50tBCSEMS6ynLEqMEKh+S6sLu0xtCFqnI8xpjH3\nrWstyK2vIBStjhMnK1GgJy9kLo+sSLDTLISz6SwALVstOyx0MQYw1KB0LSyfHK8MKX4tjKyBLBel\naLBprTAnJqvorm2k96bYnqitZC1+Lsyfo7WCq8+coKkkMhYpb6XSr0ixzKf1sLkw8TPUL/0poq4C\npqgkqrQHIyg0wDTetImcNS18MmAvrSnfIQ0ztS8IJFGxuyyPLFGdiDWTLtew4S8LLgGdXypVrD4y\neyMUMyMp0zAvM5mwKLJSIyCxgSArn5oweK7fHt+wia0WrjEsk6tMMnaoprAdJz2qjCWasNMwxqxY\nLBswBS4Wrn2zKasgsAwt+C6KsBgxyCTRKY6w16MCK5CiNreIID2mwqoHrtauiq1GLcsknaVhq5Ia\nPC4psVasXq9/L5EpVY0ysQewGjAMIYqsBC71qaMwtZwXMIyqw6hfJuSsbaYZL3eoUykrLyMqVi+m\nKAMwFCIip6m0mKw6MBewAiTSNFqsUSlPloewLqwMtEekx6twMBuq/CYQMMkwJiW4sUAoK7PnpvEv\nCyL4ryqcwi0eHJmryaw6sK4wBCnAoCmqXqZLLOSwkygMrnIZa6wkMV8utSiyMBKqDae6H88sH7Qs\nL+qkgaUiNOky6Shsr+WoILAeqI2msK+hMlgwfStWsbKpgR3VNDcqII/PrM8wO6z7rR4vdTRIqIAz\nNjRFJgIyXi/LL8GvvjADI3Al0SqAMtYZDaoStO8s1ChCo1EoQjFgr1kg8SsTpCWhcS6XrHQwdynI\nsOqnMSsqHs8pKaeLptQxEq2tMXGk2LDkrLYy16xQLq4sSzQ4NAQssLBDqiisyRTWpwYwyyucse2w\nZa8HshwjXC3vLZAo5DKjKX0vwiSQpmgs7Sk+KZUuziyyLAmso7DkMcMt2661L/CrQjFdMcyQ1yvC\nHVKqKq6eLpuvEK0wspi0Pqm0rEgpDy3RKlKvoq45qmgqeh8DHSUUn7RYLpyuOSg7NJ0v2rGxsugn\nqKhNKw2x8C+jMDip8KntIP6gJB6AMicsfBzYJCevPSrHLGAxNaMHsD+vSSx2qmQuxq12LOEqdbCq\nK4auHLQBM/qqyLHgJ9Ays5zrqhqvTqpepy0cm6yptbSpJ6AprIYpLKwcLfAq8rKEsOWjw7CzofOt\nPSqTrLyd2DAwraat1a4fKIAj0qi3rIwrljFMM02oxJ9NNNWpK6voJqexEKnrKM+yYhlfrqguEasf\noPUt/xqWsmmwhKvKK6us0asGH/Qnaq8uMD6r7q1/K7CwDKx6Kymy2qnMIcauYa6vsP2scDCkKFmv\n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\n"], "alpha/b": [[6], "BDRIsVspOyJMlnwy\n"], "dense1/b": [[64], "uyRHL/edbK5Pr4OspaocKAkszyoNLA2onigNqbya1K1mLB4oP6tzKnkkSCfFKRsk+61qJ2CruCYa\nrEklPyjPrT0tlCwvrnwrki6dp0codCjaJnirKy3FreYrNy65sHmtcayNKRAlCpq6H8cqXqlXsI+k\nTaimrtMohyq3sTSqKag=\n"]}


def LIN(x, x1, y1, x2, y2):
    return y1 + (y2-y1)*(x-x1)/(x2-x1)


MAX_THRUST = 200
NBPOD = 4
TIMEOUT = 100


class Model:
    def __init__(self):
        self.weights = {}
        for var, (shape, data) in MODEL_DATA.items():
            self.weights[var] = np.reshape(np.fromstring(base64.decodestring(data.encode()), dtype=np.float16).astype(np.float32), shape)

    def nn_predict(self, inputs):
        layer = np.array([inputs], dtype=np.float32)
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
        action = self.nn_predict(game_state.extract_state())
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
        if gene < 0.25:
            res -= 18.0
        elif gene > 0.75:
            res += 18.0
        else:
            res += LIN(gene, 0.25, -18.0, 0.75, 18.0)

        if res >= 360.0:
            res -= 360.0
        elif res < 0.0:
            res += 360.0

        return res

    def get_new_power(self, gene):
        if gene < 0.2:
            return 0
        elif gene > 0.8:
            return MAX_THRUST
        else:
            return LIN(gene, 0.2, 0, 0.8, MAX_THRUST)

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
    ai = Model()
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
