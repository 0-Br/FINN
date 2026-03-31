import numpy as np

NR = 9

def make_rain_variables2(rainlist):
    """
    based on Analysis of Hyetographs for Drainage System Modeling
    KatarzynaWartalska, Bartosz Kazmierczak, Monika Nowakowska, Andrzej Kotowski
    Water 2020, 12, 149; doi:10.3390/w12010149
    """
    rainarr = np.array(rainlist, dtype='float32')
    indpeak = np.where(rainarr == np.max(rainarr))[0][0]

    # duration
    dur = float(len(rainlist))
    # rainfall depth
    depth = np.sum(rainarr)
    # peak position ratio
    rp = (indpeak + 1) / dur
    # center of gravity position ratio (time when half of the rainfall has fallen)
    rcg = (np.min(np.where(np.cumsum(rainarr) >= (depth / 2))) + 1) / dur
    # rainfall before and after peak
    m1 = np.sum(rainarr[0: indpeak]) / np.sum(rainarr[indpeak:])
    # max intensity vs. depth
    m2 = np.max(rainarr) / depth
    # precipitation in first 33%  of rain event
    m3 = np.sum(rainarr[0: int(dur // 3)]) / depth
    # precipitation in first half of event
    m5 = np.sum(rainarr[0: int(dur // 2)]) / depth
    # maximum intensity divided by average intensity
    ni = np.max(rainarr) / np.mean(rainarr)

    return [dur, depth, rp, rcg, m1, m2, m3, m5, ni]
