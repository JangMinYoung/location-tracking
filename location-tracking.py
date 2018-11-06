#
# Our standard imports:
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
import math

# Access to many standard distributions:
import scipy.stats as ss

# x = np.random.rand(100)
# y = np.random.rand(100)
#
# # With correlate:
# # We must remove the means.
# cc1 = np.correlate(x - x.mean(), y - y.mean())[0]
# # And we must normalize by the number of points
# # and the product of the standard deviations.
# cc1 /= (len(x) * x.std() * y.std())
#
# # with corrcoef:
# cc2 = np.corrcoef(x, y)[0, 1]
#
# print(cc1, cc2)
#
# nx = 50
# x = np.random.randn(nx) # normal RV
#
# lags = np.arange(-nx + 1, nx) # so last value is nx - 1
#
# # Remove sample mean.
# xdm = x - x.mean()
#
# autocorr_xdm = np.correlate(xdm, xdm, mode='full')
# # Normalize by the zero-lag value:
# autocorr_xdm /= autocorr_xdm[nx - 1]
#
# fig, ax = plt.subplots()
# ax.plot(lags, autocorr_xdm, 'r')
# ax.set_xlabel('lag')
# ax.set_ylabel('correlation coefficient')
# ax.grid(True)
#
#
# plt.close(fig)

#
# def autocorr(x, twosided=False, tapered=True):
#     """
#     Return (lags, ac), where ac is the estimated autocorrelation
#     function for x, at the full set of possible lags.
#
#     If twosided is True, all lags will be included;
#     otherwise (default), only non-negative lags will be included.
#
#     If tapered is True (default), the low-MSE estimate, linearly
#     tapered to zero for large lags, is returned.
#     """
#     nx = len(x)
#     xdm = x - x.mean()
#     ac = np.correlate(xdm, xdm, mode='full')
#     ac /= ac[nx - 1]
#     lags = np.arange(-nx + 1, nx)
#     if not tapered:  # undo the built-in taper
#         taper = 1 - np.abs(lags) / float(nx)
#         ac /= taper
#     if twosided:
#         return lags, ac
#     else:
#         return lags[nx - 1:], ac[nx - 1:]
#
# def mean_sem_edof(y, truncated=True, tapered_cor=True):
#     """
#     Return the mean, SEM, and EDOF for the sequence y.
#
#     If truncated is True (default), the EDOF and SEM will
#     be calculated based on only the positive central peak of
#     the sample autocorrelation.
#
#     If tapered_cor is True (default), the low-MSE estimate of
#     the lagged correlation is used.
#     """
#     ym = y.mean()
#     n = len(y)
#     lags, ac = autocorr(y, twosided=True, tapered=tapered_cor)
#     taper = 1 - np.abs(lags) / n
#     if truncated:
#         i1 = np.nonzero(np.logical_and(lags >= 0, ac < 0))[0].min()
#         i0 = 2 * n - i1 - 1  # taking advantage of symmetry...
#         sl = slice(i0, i1)
#     else:
#         sl = slice(None)
#     edof = n / np.sum(taper[sl] * ac[sl])
#     with np.errstate(invalid='ignore'):
#         sem = y.std() / np.sqrt(edof)
#     return ym, sem, edof
#
# nx = 200
# t = np.arange(nx)
# x = np.random.randn(nx)
# xc = np.convolve(x, np.ones(5) / 5.0, mode='same')
#
# fig, (ax0, ax1) = plt.subplots(2)
# fig.subplots_adjust(hspace=0.4)
# ax0.plot(t, x, 'b', t, xc, 'r')
# ax0.set_xlabel('time')
# ax0.set_ylabel('random data')
#
# lags, auto_x = autocorr(x)
# lags, auto_xc = autocorr(xc)
# ax1.plot(lags, auto_x, 'b', lags, auto_xc, 'r')
# ax1.set_xlabel('lag')
# ax1.set_ylabel('correlation')
#
# for ax in (ax0, ax1):
#     ax.locator_params(axis='y', nbins=4)
# plt.close(fig)
#
# for truncated in (True, False):
#     print("Integrating over central peak? ", truncated)
#     for tapered_cor in (True, False):
#         print("  Tapered correlation estimate? ", tapered_cor)
#         print("    x:  %7.3f  %7.3f  %9.1f " %
#               mean_sem_edof(x, truncated=truncated, tapered_cor=tapered_cor))
#         print("    xc: %7.3f  %7.3f  %9.1f " %
#               mean_sem_edof(xc, truncated=truncated, tapered_cor=tapered_cor))
#         print("")

def cross_correlation(y1,y2):
    npts = len(y2)

    x = np.linspace(-3, 5, npts)
    y1=np.array(y1)
    y2=np.array(y2)

    lags = np.arange(-npts+1 , npts)
    ccov = np.correlate(y1 - y1.mean(), y2 - y2.mean(), mode='full')
    ccor = ccov / (npts * y1.std() * y2.std())

    fig, axs = plt.subplots(nrows=2)
    fig.subplots_adjust(hspace=0.4)
    ax = axs[0]
    ax.plot(x, y1, 'b', label='y1')
    ax.plot(x, y2, 'r', label='y2')
    ax.set_ylim(-5, 5)
    ax.legend(loc='upper right', fontsize='small', ncol=2)

    ax = axs[1]
    ax.plot(lags, ccor)
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel('cross-correlation')
    ax.set_xlabel('lag of y1 relative to y2')

    maxlag = lags[np.argmax(ccor)]
    print("max correlation is at lag %d" % maxlag)
    result = maxlag / 44100

    # plt.show()
    # plt.close(fig)

    return result

def cal_Altitude(a,td10,td21,td02):
    pow = math.pow(td10, 2) + math.pow(td21, 2) + math.pow(td02, 2)
    pow_sqr = math.sqrt(pow)
    constant = (math.sqrt(2) * 340) / (3 * a)
    # radian=math.radians(constant * pow_sqr)
    mul=constant * pow_sqr
    print("radian:"+str(mul))
    Altitude_angle = math.acos(mul)
    print("Altitude_angle: "+str(Altitude_angle))

    # Altitude_angle2=math.radians(86)
    # print("Altitude_angle2: " + str(Altitude_angle2))

    return Altitude_angle

def cal_Bearing(a, td21, Altitude_angle):
    theta=0
    constant=(math.sqrt(3)*a)/340
    b = 8.166603115929353e-05
    # pi_degree = math.degrees(math.pi)
    cos_altitude=math.cos(Altitude_angle)

    x=((td21/constant)/cos_altitude)
    # print("x: "+str(x))
    arcoss_value=math.acos(x)

    theta_1=arcoss_value+(math.pi/2)
    # theta_2=(math.pi/2)-arcoss_value
    # print(math.degrees(theta_2))

    if math.degrees(theta_1)>=90:
        theta=90-math.degrees(arcoss_value)
    # print(theta)
    return math.radians(theta)

def location(r,Altitude_angle,Bearing_angle):
    x=r*math.cos(Altitude_angle)*math.cos(Bearing_angle)
    y=r*math.cos(Altitude_angle)*math.sin(Bearing_angle)
    z=r*math.sin(Altitude_angle)
    return x,y,z


def main():

    a = 2
    r = 4
    y0 = [-0.05, -0.01, 0.02, 0.12, -0.26, -0.50, -0.53]
    y1 = [-0.05, 0.02, 0.12, 0.19, -0.22, -0.47, -0.50]
    y2 = [-0.05, -0.01, 0.02, 0.06, -0.22, -0.33, -0.40]

    # y0 = [-0.3,0,0.5,1.3,0.5,0,-0.3]
    # y1 = [0,0.5,1.3,0.5,0,-0.3,-0.7]
    # y2 = [0.1,1.3,0.5,0,-0.3,-0,7]

    # print("y0:"+str(len(y0)))
    # print("y1:" + str(len(y1)))
    # print("y2:" + str(len(y2))+","+str(y2))

    td10 = cross_correlation(y1,y0)
    td21 = cross_correlation(y2, y1)
    td02 = cross_correlation(y0, y2)

    Altitude_angle = cal_Altitude(a,td10,td21,td02)
    Bearing_angle = cal_Bearing(a, td21, Altitude_angle)
    x,y,z = location(r, Altitude_angle, Bearing_angle)

    print("x:"+str(x),"y:"+str(y),"z:"+str(z))

main()