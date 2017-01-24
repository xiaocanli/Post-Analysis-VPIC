
from matplotlib.colors import LinearSegmentedColormap
from numpy import inf, nan

# Used to reconstruct the colormap in viscm
parameters = {'xp': [28.782590300914933, 60.468890475434989, -24.282418425088565, -47.188177587392218, -34.590010048125208, -8.6301496641810616],
              'yp': [16.928446771378731, -42.626527050610804, -74.312827225130874, -5.5955497382198871, 42.5065445026178, 39.070680628272271],
              'min_JK': 18.0859375,
              'max_JK': 95.0390625}

cm_data = [[ 0.30593816,  0.00266902,  0.0061051 ],
       [ 0.31049445,  0.00281314,  0.01074331],
       [ 0.31501841,  0.00297468,  0.01602065],
       [ 0.3195184 ,  0.00312249,  0.02204556],
       [ 0.32398709,  0.00328294,  0.02880862],
       [ 0.32843006,  0.00343291,  0.03641347],
       [ 0.33284287,  0.00358919,  0.04466259],
       [ 0.33722794,  0.00373997,  0.05287228],
       [ 0.34158401,  0.00388914,  0.06102198],
       [ 0.34591015,  0.0040397 ,  0.06913597],
       [ 0.35020835,  0.00417912,  0.07726788],
       [ 0.35447435,  0.00432876,  0.08538353],
       [ 0.35871326,  0.00445639,  0.09357214],
       [ 0.36291947,  0.00459302,  0.10177173],
       [ 0.36709555,  0.00471965,  0.11003439],
       [ 0.37124003,  0.00484227,  0.11835335],
       [ 0.37535139,  0.00496951,  0.12671589],
       [ 0.37943182,  0.00507827,  0.13517718],
       [ 0.38347763,  0.0051954 ,  0.1436843 ],
       [ 0.38748989,  0.00530563,  0.15227436],
       [ 0.39146758,  0.00541166,  0.16094589],
       [ 0.39540886,  0.00552783,  0.169674  ],
       [ 0.3993146 ,  0.00563205,  0.17850829],
       [ 0.40318253,  0.00574449,  0.18741263],
       [ 0.40701159,  0.00587141,  0.19638009],
       [ 0.41080188,  0.00598896,  0.20546335],
       [ 0.41455113,  0.0061258 ,  0.21461195],
       [ 0.41825834,  0.00628491,  0.22382709],
       [ 0.42192262,  0.00644595,  0.23315539],
       [ 0.42554221,  0.0066368 ,  0.24255105],
       [ 0.42911584,  0.00686223,  0.25201397],
       [ 0.43264184,  0.00711059,  0.26157606],
       [ 0.43611865,  0.00739991,  0.27121371],
       [ 0.43954483,  0.00774183,  0.28091667],
       [ 0.44291849,  0.0081382 ,  0.29069433],
       [ 0.44623724,  0.00858895,  0.30056102],
       [ 0.4494999 ,  0.00911734,  0.31048822],
       [ 0.45270459,  0.00973307,  0.32047446],
       [ 0.45584867,  0.0104402 ,  0.33053061],
       [ 0.45892961,  0.01124886,  0.34065775],
       [ 0.46194612,  0.01218144,  0.35083506],
       [ 0.46489594,  0.01325186,  0.3610596 ],
       [ 0.46777666,  0.01447465,  0.37132922],
       [ 0.47058405,  0.01585603,  0.38166101],
       [ 0.47331737,  0.01742581,  0.39202674],
       [ 0.47597409,  0.01920295,  0.40242192],
       [ 0.47855157,  0.02120777,  0.41284165],
       [ 0.48104715,  0.02346196,  0.42328061],
       [ 0.48345624,  0.02598275,  0.43374728],
       [ 0.48577749,  0.02880089,  0.44422203],
       [ 0.48800839,  0.03194345,  0.45469593],
       [ 0.49014605,  0.03543786,  0.46516216],
       [ 0.49218758,  0.03931295,  0.47561354],
       [ 0.49413003,  0.04347853,  0.48604251],
       [ 0.49597027,  0.04779536,  0.49644208],
       [ 0.49770377,  0.05226892,  0.50681131],
       [ 0.49932935,  0.05689842,  0.51713227],
       [ 0.50084406,  0.06168052,  0.52739632],
       [ 0.50224498,  0.06661185,  0.53759456],
       [ 0.50352922,  0.07168902,  0.54771787],
       [ 0.50469393,  0.07690861,  0.55775699],
       [ 0.50573632,  0.0822671 ,  0.56770249],
       [ 0.50665366,  0.0877609 ,  0.57754487],
       [ 0.50744331,  0.09338632,  0.58727454],
       [ 0.5081027 ,  0.09913955,  0.59688191],
       [ 0.50862938,  0.10501665,  0.6063574 ],
       [ 0.50902102,  0.11101356,  0.61569147],
       [ 0.5092751 ,  0.11712627,  0.62487537],
       [ 0.50938958,  0.12335046,  0.63389967],
       [ 0.50936297,  0.12968134,  0.64275439],
       [ 0.50919354,  0.13611425,  0.65143072],
       [ 0.50887974,  0.1426444 ,  0.65992011],
       [ 0.50842022,  0.14926685,  0.66821431],
       [ 0.50781383,  0.15597655,  0.67630542],
       [ 0.50705966,  0.16276831,  0.68418591],
       [ 0.50615699,  0.16963687,  0.69184866],
       [ 0.50510539,  0.17657683,  0.69928701],
       [ 0.50390464,  0.18358274,  0.70649474],
       [ 0.5025548 ,  0.19064905,  0.71346616],
       [ 0.50105621,  0.19777018,  0.72019609],
       [ 0.49940946,  0.2049405 ,  0.72667991],
       [ 0.49761526,  0.2121545 ,  0.73291365],
       [ 0.49567458,  0.21940666,  0.73889392],
       [ 0.49358924,  0.22669104,  0.74461755],
       [ 0.49136101,  0.234002  ,  0.75008223],
       [ 0.48899197,  0.24133394,  0.75528627],
       [ 0.48648446,  0.24868133,  0.76022859],
       [ 0.48384116,  0.2560387 ,  0.76490869],
       [ 0.48106503,  0.26340071,  0.7693267 ],
       [ 0.4781593 ,  0.27076208,  0.77348331],
       [ 0.47512752,  0.2781177 ,  0.77737983],
       [ 0.4719735 ,  0.28546258,  0.78101811],
       [ 0.46870133,  0.29279188,  0.78440055],
       [ 0.46531539,  0.30010094,  0.78753008],
       [ 0.46182028,  0.30738527,  0.79041016],
       [ 0.45822089,  0.31464057,  0.7930447 ],
       [ 0.45452231,  0.32186273,  0.79543808],
       [ 0.45072988,  0.32904785,  0.79759511],
       [ 0.44684916,  0.33619226,  0.79952102],
       [ 0.44288567,  0.34329262,  0.80122132],
       [ 0.43884568,  0.3503455 ,  0.802702  ],
       [ 0.43473532,  0.35734787,  0.80396929],
       [ 0.43056086,  0.36429696,  0.80502972],
       [ 0.42632872,  0.37119022,  0.80589002],
       [ 0.42204544,  0.37802533,  0.80655715],
       [ 0.41771766,  0.38480021,  0.80703824],
       [ 0.41335213,  0.39151301,  0.80734056],
       [ 0.40895566,  0.39816209,  0.8074715 ],
       [ 0.40453513,  0.40474605,  0.8074385 ],
       [ 0.40009746,  0.4112637 ,  0.8072491 ],
       [ 0.39564958,  0.41771406,  0.80691081],
       [ 0.39119843,  0.42409636,  0.80643115],
       [ 0.38675107,  0.43040998,  0.8058177 ],
       [ 0.3823147 ,  0.43665438,  0.80507805],
       [ 0.37789581,  0.44282955,  0.80421938],
       [ 0.37350115,  0.44893549,  0.80324892],
       [ 0.3691374 ,  0.45497233,  0.80217381],
       [ 0.36481112,  0.46094041,  0.80100101],
       [ 0.36052875,  0.46684018,  0.79973734],
       [ 0.35629656,  0.47267226,  0.79838945],
       [ 0.35212066,  0.47843739,  0.79696377],
       [ 0.34800697,  0.48413643,  0.79546656],
       [ 0.34396121,  0.48977038,  0.79390384],
       [ 0.33998883,  0.49534031,  0.79228142],
       [ 0.33609505,  0.50084744,  0.79060487],
       [ 0.33228481,  0.50629304,  0.7888795 ],
       [ 0.32856276,  0.51167848,  0.78711041],
       [ 0.32493321,  0.51700519,  0.78530242],
       [ 0.32140017,  0.52227471,  0.78346009],
       [ 0.31796726,  0.52748859,  0.78158772],
       [ 0.31463776,  0.53264846,  0.77968936],
       [ 0.31141453,  0.537756  ,  0.77776877],
       [ 0.30830029,  0.54281285,  0.77582965],
       [ 0.30529701,  0.54782078,  0.77387515],
       [ 0.30240618,  0.55278164,  0.77190811],
       [ 0.299629  ,  0.55769722,  0.76993121],
       [ 0.29696628,  0.56256933,  0.76794687],
       [ 0.29441839,  0.5673998 ,  0.76595727],
       [ 0.29198527,  0.57219047,  0.76396427],
       [ 0.28966643,  0.57694317,  0.76196953],
       [ 0.28746101,  0.58165974,  0.75997439],
       [ 0.28536773,  0.586342  ,  0.75797996],
       [ 0.28338495,  0.59099176,  0.7559871 ],
       [ 0.28151069,  0.59561083,  0.7539964 ],
       [ 0.27974263,  0.60020097,  0.7520082 ],
       [ 0.27807817,  0.60476394,  0.75002261],
       [ 0.27651446,  0.60930147,  0.74803947],
       [ 0.27504843,  0.61381525,  0.74605839],
       [ 0.27367681,  0.61830693,  0.74407875],
       [ 0.27239621,  0.62277814,  0.74209971],
       [ 0.27120315,  0.62723044,  0.74012017],
       [ 0.2700941 ,  0.63166536,  0.73813883],
       [ 0.26906553,  0.63608438,  0.73615419],
       [ 0.26811396,  0.64048893,  0.7341645 ],
       [ 0.267236  ,  0.64488036,  0.73216784],
       [ 0.26642845,  0.64925999,  0.73016208],
       [ 0.26568825,  0.65362905,  0.7281449 ],
       [ 0.26501263,  0.65798873,  0.72611382],
       [ 0.2643991 ,  0.66234012,  0.72406616],
       [ 0.26384553,  0.66668426,  0.72199908],
       [ 0.26335016,  0.67102211,  0.7199096 ],
       [ 0.26291166,  0.67535454,  0.7177946 ],
       [ 0.26252925,  0.67968235,  0.71565087],
       [ 0.26220252,  0.68400627,  0.71347494],
       [ 0.26193168,  0.68832694,  0.71126329],
       [ 0.26171752,  0.6926449 ,  0.70901233],
       [ 0.26156145,  0.69696062,  0.70671835],
       [ 0.26146552,  0.70127446,  0.70437758],
       [ 0.26143241,  0.70558672,  0.70198618],
       [ 0.26146552,  0.70989759,  0.69954026],
       [ 0.26156889,  0.71420716,  0.69703589],
       [ 0.26174728,  0.71851546,  0.6944691 ],
       [ 0.26200613,  0.72282241,  0.69183591],
       [ 0.26235155,  0.72712783,  0.68913235],
       [ 0.26279032,  0.73143149,  0.68635442],
       [ 0.26332987,  0.73573302,  0.68349817],
       [ 0.26397823,  0.74003201,  0.68055967],
       [ 0.26474399,  0.74432795,  0.67753502],
       [ 0.26563628,  0.74862022,  0.67442037],
       [ 0.26666467,  0.75290817,  0.67121195],
       [ 0.26783915,  0.75719102,  0.66790602],
       [ 0.26917   ,  0.76146795,  0.66449895],
       [ 0.27066777,  0.76573805,  0.66098717],
       [ 0.27234313,  0.77000035,  0.6573672 ],
       [ 0.27420683,  0.77425378,  0.65363566],
       [ 0.27626956,  0.77849724,  0.64978928],
       [ 0.2785419 ,  0.78272954,  0.64582489],
       [ 0.28103415,  0.78694946,  0.64173941],
       [ 0.2837563 ,  0.79115568,  0.63752989],
       [ 0.28671789,  0.79534685,  0.63319351],
       [ 0.28992795,  0.79952157,  0.62872753],
       [ 0.29339489,  0.80367837,  0.62412938],
       [ 0.29712646,  0.80781575,  0.61939656],
       [ 0.30112967,  0.81193215,  0.61452672],
       [ 0.30541076,  0.81602596,  0.60951765],
       [ 0.30997513,  0.82009555,  0.60436723],
       [ 0.31482736,  0.82413923,  0.59907348],
       [ 0.31997118,  0.82815528,  0.59363455],
       [ 0.32540947,  0.83214193,  0.58804873],
       [ 0.33114429,  0.83609738,  0.58231439],
       [ 0.33717691,  0.84001981,  0.57643009],
       [ 0.3435078 ,  0.84390735,  0.57039447],
       [ 0.35013676,  0.84775809,  0.56420634],
       [ 0.35706288,  0.8515701 ,  0.55786461],
       [ 0.36428465,  0.85534142,  0.55136836],
       [ 0.37179997,  0.85907004,  0.5447168 ],
       [ 0.37960764,  0.86275396,  0.53790608],
       [ 0.38770397,  0.86639106,  0.53093767],
       [ 0.39608533,  0.86997925,  0.52381182],
       [ 0.404748  ,  0.8735164 ,  0.51652834],
       [ 0.41368796,  0.87700037,  0.50908724],
       [ 0.42290092,  0.88042896,  0.50148873],
       [ 0.43238239,  0.88379999,  0.49373327],
       [ 0.44212768,  0.88711122,  0.48582155],
       [ 0.45213194,  0.89036041,  0.47775457],
       [ 0.46239249,  0.89354509,  0.46952983],
       [ 0.47290661,  0.8966627 ,  0.4611454 ],
       [ 0.48366544,  0.89971121,  0.45260956],
       [ 0.4946637 ,  0.90268831,  0.44392485],
       [ 0.50589594,  0.90559166,  0.43509433],
       [ 0.5173566 ,  0.90841895,  0.42612172],
       [ 0.52904678,  0.911167  ,  0.4170014 ],
       [ 0.54095942,  0.91383355,  0.40774077],
       [ 0.55308405,  0.91641689,  0.39835269],
       [ 0.56541433,  0.91891481,  0.38884459],
       [ 0.57794756,  0.92132456,  0.37921977],
       [ 0.59068532,  0.92364261,  0.36947702],
       [ 0.60360889,  0.92586874,  0.3596447 ],
       [ 0.61671039,  0.9280012 ,  0.34973737],
       [ 0.6299949 ,  0.93003583,  0.33975421],
       [ 0.64344639,  0.93197236,  0.32972537],
       [ 0.65704942,  0.93381072,  0.31968252],
       [ 0.67080705,  0.93554723,  0.3096364 ],
       [ 0.68470183,  0.9371825 ,  0.29962866],
       [ 0.69871405,  0.93871806,  0.28970807],
       [ 0.71284969,  0.94014982,  0.2798966 ],
       [ 0.72707567,  0.94148299,  0.27027164],
       [ 0.74138372,  0.94271735,  0.26088676],
       [ 0.7557575 ,  0.943855  ,  0.25181386],
       [ 0.77017105,  0.94490104,  0.24314461],
       [ 0.78461467,  0.94585678,  0.23496111],
       [ 0.79905507,  0.94673057,  0.2273802 ],
       [ 0.81347933,  0.94752572,  0.22050384],
       [ 0.82785668,  0.94825138,  0.21445687],
       [ 0.8421664 ,  0.94891453,  0.20935427],
       [ 0.85638392,  0.94952403,  0.2053095 ],
       [ 0.87048425,  0.95008971,  0.20242398],
       [ 0.88444705,  0.95062061,  0.20077676],
       [ 0.89824938,  0.9511274 ,  0.200423  ],
       [ 0.91187192,  0.95162019,  0.20138551],
       [ 0.92530211,  0.95210721,  0.20365371],
       [ 0.93851494,  0.95260192,  0.20718834],
       [ 0.95151447,  0.9531072 ,  0.21192152],
       [ 0.9642739 ,  0.95363819,  0.21776488],
       [ 0.9767993 ,  0.95419746,  0.22461997],
       [ 0.98909212,  0.95478923,  0.2323852 ]]

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
