
from matplotlib.colors import LinearSegmentedColormap
from numpy import inf, nan

# Used to reconstruct the colormap in viscm
parameters = {'xp': [22.674387857633945, 11.221508276482126, -14.356589454756971, -47.188177587392218, -34.590010048125208, 0.15039134803535603],
              'yp': [-20.102530541012214, -33.082460732984288, -42.24476439790574, -5.5955497382198871, 42.5065445026178, 24.563699825479944],
              'min_JK': 18.8671875,
              'max_JK': 92.5}

cm_data = [[ 0.26700401,  0.00487433,  0.32941519],
       [ 0.2685542 ,  0.00957471,  0.33533275],
       [ 0.27003443,  0.01455696,  0.34119436],
       [ 0.27144484,  0.01982836,  0.34699773],
       [ 0.27278686,  0.02539504,  0.35273831],
       [ 0.27405872,  0.03126572,  0.35841766],
       [ 0.27526033,  0.03744784,  0.36403393],
       [ 0.27639307,  0.04380557,  0.36958299],
       [ 0.2774554 ,  0.04993149,  0.37506557],
       [ 0.27844714,  0.05585994,  0.38048004],
       [ 0.27936893,  0.06162751,  0.38582368],
       [ 0.28022049,  0.0672627 ,  0.39109518],
       [ 0.28100114,  0.07278804,  0.39629373],
       [ 0.28171081,  0.07822095,  0.40141769],
       [ 0.28235053,  0.0835745 ,  0.40646432],
       [ 0.28291925,  0.08886094,  0.4114333 ],
       [ 0.2834169 ,  0.09408964,  0.41632321],
       [ 0.28384358,  0.09926827,  0.42113253],
       [ 0.28420026,  0.10440257,  0.42585912],
       [ 0.28448608,  0.10949867,  0.43050257],
       [ 0.2847012 ,  0.11456108,  0.43506157],
       [ 0.28484581,  0.11959358,  0.43953491],
       [ 0.28492117,  0.12459863,  0.44392086],
       [ 0.28492661,  0.12957963,  0.44821903],
       [ 0.28486252,  0.1345388 ,  0.45242844],
       [ 0.28472931,  0.13947799,  0.45654822],
       [ 0.28452817,  0.14439827,  0.46057726],
       [ 0.28425933,  0.14930113,  0.46451508],
       [ 0.28392311,  0.15418778,  0.46836123],
       [ 0.28352022,  0.15905901,  0.47211522],
       [ 0.28305157,  0.16391535,  0.47577663],
       [ 0.28251884,  0.16875671,  0.479345  ],
       [ 0.28192206,  0.17358397,  0.48282047],
       [ 0.28126222,  0.17839731,  0.486203  ],
       [ 0.28054043,  0.18319677,  0.48949266],
       [ 0.27975811,  0.18798216,  0.49268964],
       [ 0.27891713,  0.19275301,  0.49579421],
       [ 0.27801794,  0.19750965,  0.49880695],
       [ 0.27706191,  0.20225183,  0.50172842],
       [ 0.27605049,  0.20697927,  0.50455929],
       [ 0.27498517,  0.21169165,  0.50730037],
       [ 0.27386887,  0.21638789,  0.50995262],
       [ 0.272702  ,  0.22106829,  0.51251706],
       [ 0.27148621,  0.22573247,  0.51499482],
       [ 0.27022324,  0.23038004,  0.51738712],
       [ 0.26891491,  0.23501057,  0.51969528],
       [ 0.26756319,  0.2396236 ,  0.52192074],
       [ 0.26617119,  0.24421813,  0.52406522],
       [ 0.26473951,  0.24879442,  0.52613012],
       [ 0.26327012,  0.25335212,  0.52811709],
       [ 0.26176496,  0.25789086,  0.53002785],
       [ 0.26022606,  0.26241031,  0.53186418],
       [ 0.25865541,  0.26691017,  0.5336279 ],
       [ 0.25705544,  0.27138997,  0.53532102],
       [ 0.25542898,  0.27584915,  0.53694571],
       [ 0.25377689,  0.28028798,  0.5385036 ],
       [ 0.25210122,  0.28470627,  0.53999668],
       [ 0.25040396,  0.28910388,  0.54142695],
       [ 0.24868713,  0.29348068,  0.54279642],
       [ 0.2469527 ,  0.29783657,  0.54410709],
       [ 0.24520266,  0.3021715 ,  0.545361  ],
       [ 0.24343892,  0.30648544,  0.54656016],
       [ 0.24166391,  0.31077823,  0.54770676],
       [ 0.23987982,  0.31504978,  0.54880295],
       [ 0.23808762,  0.3193005 ,  0.54985035],
       [ 0.23628909,  0.3235305 ,  0.5508509 ],
       [ 0.23448595,  0.32773993,  0.5518065 ],
       [ 0.23267989,  0.33192894,  0.55271901],
       [ 0.23087254,  0.33609773,  0.55359027],
       [ 0.22906545,  0.34024654,  0.55442204],
       [ 0.22726014,  0.34437561,  0.55521608],
       [ 0.22545806,  0.34848521,  0.55597407],
       [ 0.22366058,  0.35257566,  0.55669764],
       [ 0.22186902,  0.35664726,  0.55738837],
       [ 0.22008461,  0.36070037,  0.55804779],
       [ 0.21830854,  0.36473534,  0.55867737],
       [ 0.21654188,  0.36875256,  0.55927849],
       [ 0.21478568,  0.37275242,  0.55985251],
       [ 0.21304088,  0.37673533,  0.5604007 ],
       [ 0.21130836,  0.38070172,  0.56092426],
       [ 0.20958891,  0.38465203,  0.56142436],
       [ 0.20788325,  0.3885867 ,  0.56190206],
       [ 0.20619204,  0.3925062 ,  0.56235839],
       [ 0.20451584,  0.39641098,  0.56279428],
       [ 0.20285517,  0.40030153,  0.56321061],
       [ 0.20121044,  0.40417832,  0.56360819],
       [ 0.19958202,  0.40804183,  0.56398778],
       [ 0.19797019,  0.41189255,  0.56435004],
       [ 0.1963752 ,  0.41573098,  0.56469559],
       [ 0.19479719,  0.41955759,  0.56502496],
       [ 0.19323628,  0.42337288,  0.56533864],
       [ 0.19169253,  0.42717733,  0.56563705],
       [ 0.19016593,  0.43097144,  0.56592052],
       [ 0.18865644,  0.43475567,  0.56618935],
       [ 0.18716399,  0.43853052,  0.56644376],
       [ 0.18568845,  0.44229644,  0.56668391],
       [ 0.1842297 ,  0.44605392,  0.5669099 ],
       [ 0.18278756,  0.4498034 ,  0.56712177],
       [ 0.18136186,  0.45354533,  0.56731952],
       [ 0.17995242,  0.45728016,  0.56750306],
       [ 0.17855907,  0.46100832,  0.56767227],
       [ 0.17718163,  0.46473023,  0.56782696],
       [ 0.17581998,  0.4684463 ,  0.5679669 ],
       [ 0.17447399,  0.47215693,  0.5680918 ],
       [ 0.17314359,  0.4758625 ,  0.56820133],
       [ 0.17182876,  0.47956339,  0.5682951 ],
       [ 0.17052956,  0.48325995,  0.56837267],
       [ 0.1692461 ,  0.48695252,  0.56843358],
       [ 0.16797885,  0.49064139,  0.56847745],
       [ 0.16672853,  0.49432679,  0.56850396],
       [ 0.16549485,  0.49800916,  0.5685121 ],
       [ 0.16427838,  0.50168876,  0.56850123],
       [ 0.16307978,  0.50536586,  0.56847067],
       [ 0.16189991,  0.50904071,  0.56841971],
       [ 0.16073977,  0.51271352,  0.5683476 ],
       [ 0.15960057,  0.51638451,  0.56825356],
       [ 0.1584837 ,  0.52005386,  0.56813679],
       [ 0.15739078,  0.52372174,  0.56799646],
       [ 0.15632364,  0.5273883 ,  0.56783172],
       [ 0.15528438,  0.53105366,  0.56764168],
       [ 0.15427564,  0.53471788,  0.56742564],
       [ 0.15330014,  0.53838103,  0.56718273],
       [ 0.15236029,  0.54204327,  0.56691174],
       [ 0.1514593 ,  0.54570462,  0.56661171],
       [ 0.15060068,  0.54936512,  0.56628168],
       [ 0.14978824,  0.55302477,  0.56592069],
       [ 0.14902614,  0.55668353,  0.56552775],
       [ 0.14831885,  0.56034138,  0.56510188],
       [ 0.14767117,  0.56399824,  0.5646421 ],
       [ 0.14708824,  0.56765403,  0.56414742],
       [ 0.14657555,  0.57130864,  0.56361689],
       [ 0.14613895,  0.57496191,  0.5630496 ],
       [ 0.14578423,  0.57861374,  0.56244438],
       [ 0.14551771,  0.58226396,  0.56180026],
       [ 0.14534595,  0.58591236,  0.56111627],
       [ 0.14527574,  0.58955873,  0.56039146],
       [ 0.14531405,  0.59320283,  0.55962489],
       [ 0.14546796,  0.5968444 ,  0.55881564],
       [ 0.14574467,  0.60048316,  0.5579628 ],
       [ 0.14615138,  0.60411879,  0.55706548],
       [ 0.14669517,  0.607751  ,  0.55612273],
       [ 0.14738303,  0.61137946,  0.55513357],
       [ 0.14822204,  0.61500377,  0.5540973 ],
       [ 0.14921896,  0.61862356,  0.55301312],
       [ 0.15038025,  0.6222384 ,  0.55188028],
       [ 0.15171204,  0.62584787,  0.55069804],
       [ 0.15322007,  0.62945152,  0.54946571],
       [ 0.15490964,  0.63304888,  0.54818262],
       [ 0.15678555,  0.63663945,  0.54684816],
       [ 0.15885206,  0.64022273,  0.54546173],
       [ 0.16111284,  0.6437982 ,  0.54402268],
       [ 0.16357079,  0.64736537,  0.54252996],
       [ 0.16622882,  0.6509236 ,  0.54098368],
       [ 0.16908891,  0.65447229,  0.53938344],
       [ 0.17215248,  0.65801084,  0.53772891],
       [ 0.1754204 ,  0.66153862,  0.53601978],
       [ 0.17889298,  0.66505497,  0.53425584],
       [ 0.18257005,  0.66855923,  0.53243691],
       [ 0.18645098,  0.67205072,  0.53056289],
       [ 0.19053468,  0.67552871,  0.52863375],
       [ 0.19481972,  0.6789925 ,  0.52664953],
       [ 0.1993043 ,  0.68244133,  0.52461034],
       [ 0.20398632,  0.68587445,  0.52251639],
       [ 0.20886365,  0.68929113,  0.52036688],
       [ 0.21393364,  0.69269051,  0.5181632 ],
       [ 0.21919356,  0.69607177,  0.51590585],
       [ 0.22464052,  0.69943407,  0.51359544],
       [ 0.23027153,  0.70277656,  0.51123265],
       [ 0.23608352,  0.70609837,  0.50881832],
       [ 0.24207335,  0.70939862,  0.50635339],
       [ 0.24823786,  0.71267639,  0.50383895],
       [ 0.25457385,  0.71593078,  0.50127621],
       [ 0.26107813,  0.71916085,  0.49866655],
       [ 0.2677475 ,  0.72236566,  0.49601153],
       [ 0.27457878,  0.72554425,  0.49331285],
       [ 0.2815688 ,  0.72869565,  0.49057242],
       [ 0.2887144 ,  0.73181887,  0.48779233],
       [ 0.29601244,  0.73491295,  0.48497491],
       [ 0.30345981,  0.73797687,  0.48212268],
       [ 0.31105336,  0.74100964,  0.47923843],
       [ 0.31878998,  0.74401027,  0.47632519],
       [ 0.32666653,  0.74697776,  0.47338625],
       [ 0.33467983,  0.74991112,  0.47042521],
       [ 0.34282668,  0.75280936,  0.46744595],
       [ 0.35110384,  0.75567153,  0.46445269],
       [ 0.35950797,  0.75849666,  0.46144996],
       [ 0.36803567,  0.76128384,  0.45844266],
       [ 0.37668344,  0.76403218,  0.45543604],
       [ 0.38544765,  0.7667408 ,  0.45243575],
       [ 0.39432454,  0.7694089 ,  0.44944779],
       [ 0.40331021,  0.77203572,  0.44647859],
       [ 0.41240057,  0.77462055,  0.44353498],
       [ 0.42159135,  0.77716276,  0.4406242 ],
       [ 0.43087809,  0.77966179,  0.43775387],
       [ 0.4402561 ,  0.78211718,  0.43493205],
       [ 0.44972048,  0.78452855,  0.43216717],
       [ 0.45926607,  0.78689563,  0.42946802],
       [ 0.46888748,  0.7892183 ,  0.42684375],
       [ 0.47857908,  0.79149652,  0.42430383],
       [ 0.48833499,  0.79373043,  0.42185799],
       [ 0.49814908,  0.79592027,  0.41951618],
       [ 0.50801501,  0.79806648,  0.41728852],
       [ 0.51792621,  0.80016963,  0.41518523],
       [ 0.5278759 ,  0.80223047,  0.41321655],
       [ 0.53785667,  0.80425   ,  0.41139298],
       [ 0.5478616 ,  0.8062293 ,  0.40972442],
       [ 0.55788351,  0.8081696 ,  0.40822062],
       [ 0.56791519,  0.81007234,  0.40689099],
       [ 0.57794942,  0.81193908,  0.40574446],
       [ 0.58797905,  0.81377155,  0.40478944],
       [ 0.59799701,  0.81557158,  0.4040337 ],
       [ 0.60799641,  0.81734115,  0.40348428],
       [ 0.61797054,  0.81908233,  0.40314744],
       [ 0.62791297,  0.82079727,  0.40302856],
       [ 0.63781758,  0.8224882 ,  0.40313214],
       [ 0.6476786 ,  0.82415737,  0.40346171],
       [ 0.65749067,  0.82580707,  0.40401987],
       [ 0.66724885,  0.82743959,  0.40480824],
       [ 0.67694867,  0.82905718,  0.40582751],
       [ 0.68658614,  0.83066207,  0.40707747],
       [ 0.69615776,  0.83225641,  0.40855703],
       [ 0.70566056,  0.83384229,  0.41026431],
       [ 0.71509205,  0.8354217 ,  0.41219669],
       [ 0.72445026,  0.83699653,  0.41435088],
       [ 0.73373341,  0.83856862,  0.41672307],
       [ 0.74293191,  0.84014248,  0.41930934],
       [ 0.75205237,  0.84171735,  0.42210392],
       [ 0.76109459,  0.84329463,  0.42510159],
       [ 0.77005877,  0.84487561,  0.4282969 ],
       [ 0.7789455 ,  0.84646142,  0.43168424],
       [ 0.7877557 ,  0.84805303,  0.43525793],
       [ 0.79649058,  0.84965132,  0.43901227],
       [ 0.80513957,  0.85126153,  0.44293939],
       [ 0.81370767,  0.85288321,  0.44703322],
       [ 0.82220381,  0.85451429,  0.4512893 ],
       [ 0.83063017,  0.85615512,  0.45570229],
       [ 0.83898905,  0.85780596,  0.4602671 ],
       [ 0.84726348,  0.85947482,  0.46497193],
       [ 0.85546684,  0.86115762,  0.46981455],
       [ 0.86360871,  0.86285158,  0.47479281],
       [ 0.87169181,  0.86455657,  0.47990267],
       [ 0.87969472,  0.86628276,  0.48512849],
       [ 0.88763554,  0.86802364,  0.49047254],
       [ 0.89552455,  0.86977573,  0.49593512],
       [ 0.90335389,  0.8715434 ,  0.50150708],
       [ 0.91111136,  0.87333317,  0.50717547],
       [ 0.91882427,  0.87513377,  0.51295184],
       [ 0.92648868,  0.87694776,  0.51882966],
       [ 0.93408127,  0.87878692,  0.52478777],
       [ 0.94163648,  0.88063609,  0.53084585],
       [ 0.94914523,  0.88250029,  0.5369936 ],
       [ 0.95659136,  0.88438815,  0.54321369],
       [ 0.964007  ,  0.88628495,  0.54952816],
       [ 0.97136951,  0.88820246,  0.55591444],
       [ 0.97868693,  0.89013755,  0.56237537],
       [ 0.98598024,  0.89208031,  0.56892728]]

test_cm = LinearSegmentedColormap.from_list(__file__, cm_data)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        from viscm import viscm
        viscm(test_cm)
    except ImportError:
        print("viscm not found, falling back on simple display")
        plt.imshow(np.linspace(0, 100, 256)[None, :], aspect='auto',
                   cmap=test_cm)
    plt.show()
