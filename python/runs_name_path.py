"""
Name and path for different runs
"""


def ApJ_long_paper_runs():
    """Runs for ApJ long paper
    """
    base_dirs = []
    base_dirs.append('/net/scratch2/xiaocanli/mime25-sigma01-beta02-200-100/')
    base_dirs.append(
        '/net/scratch2/xiaocanli/mime25-sigma033-beta006-200-100/')
    base_dirs.append('/net/scratch2/xiaocanli/sigma1-mime25-beta001/')
    # base_dirs.append('/scratch3/xiaocanli/sigma1-mime25-beta0003-npc200/')
    base_dirs.append('/net/scratch2/xiaocanli/mime25-guide0-beta0007-200-100/')
    base_dirs.append('/net/scratch2/xiaocanli/sigma1-mime100-beta001-mustang/')
    base_dirs.append('/net/scratch2/xiaocanli/mime25-guide0-beta001-200-100/')
    base_dirs.append(
        '/net/scratch2/xiaocanli/mime25-guide0-beta001-200-100-sigma033/')
    base_dirs.append(
        '/net/scratch2/xiaocanli/mime25-sigma1-beta002-200-100-noperturb/')
    base_dirs.append('/net/scratch2/guofan/sigma1-mime25-beta001-track-3/')
    run_names = []
    run_names.append('mime25_beta02')
    run_names.append('mime25_beta007')
    run_names.append('mime25_beta002')
    run_names.append('mime25_beta0007')
    run_names.append('mime100_beta002')
    run_names.append('mime25_beta002_sigma01')
    run_names.append('mime25_beta002_sigma033')
    run_names.append('mime25_beta002_noperturb')
    run_names.append('mime25_beta002_track')
    return (base_dirs, run_names)


def guide_field_runs():
    fname_head = 'mime25-sigma1-beta002-guide'
    base_dirs = []
    base_dirs.append('/net/scratch2/xiaocanli/sigma1-mime25-beta001/')
    fname = '/net/scratch3/xiaocanli/' + fname_head + '02-200-100/'
    base_dirs.append(fname)
    fname = '/net/scratch3/xiaocanli/' + fname_head + '05-200-100/'
    base_dirs.append(fname)
    fname = '/net/scratch2/xiaocanli/' + fname_head + '1-200-100/'
    base_dirs.append(fname)
    fname = '/net/scratch2/xiaocanli/' + fname_head + '4-200-100/'
    base_dirs.append(fname)
    run_names = []
    run_names.append('mime25_beta002')
    run_names.append('mime25_beta002_guide02')
    run_names.append('mime25_beta002_guide05')
    run_names.append('mime25_beta002_guide1')
    run_names.append('mime25_beta002_guide4')

    return (base_dirs, run_names)


def high_sigma_runs():
    fname_head = 'mime25-sigma1-beta002-guide'
    base_dirs = []
    fname = '/net/scratch3/xiaocanli/mime25-sigma30-200-100/'
    base_dirs.append(fname)
    fname = '/net/scratch3/xiaocanli/mime25-sigma100-200-100/'
    base_dirs.append(fname)
    run_names = []
    run_names.append('mime25_sigma30')
    run_names.append('mime25_sigma100')

    return (base_dirs, run_names)


def shock_sheet_runs():
    base_dirs = []
    fname = '/net/scratch3/xiaocanli/2D-90-Mach4-sheet4-multi/'
    base_dirs.append(fname)
    run_names = []
    run_names.append('2D-90-Mach4-sheet4-multi')

    return (base_dirs, run_names)


def low_beta_runs():
    base_dirs = []
    fname = '/net/scratch2/guofan/sigma1-mime25-beta0001/'
    base_dirs.append(fname)
    fname = '/net/scratch2/guofan/sigma1-mime25-beta0002-1127/'
    base_dirs.append(fname)
    run_names = []
    run_names.append('mime25_beta0001')
    run_names.append('sigma1-mime25-beta0002')

    return (base_dirs, run_names)


if __name__ == "__main__":
    ApJ_long_paper_runs()
