"""
Name and path for different runs
"""

def ApJ_long_paper_runs():
    """Runs for ApJ long paper
    """
    base_dirs = []
    base_dirs.append('/net/scratch2/xiaocanli/mime25-sigma01-beta02-200-100/')
    base_dirs.append('/net/scratch2/xiaocanli/mime25-sigma033-beta006-200-100/')
    base_dirs.append('/scratch3/xiaocanli/sigma1-mime25-beta001/')
    base_dirs.append('/scratch3/xiaocanli/sigma1-mime25-beta0003-npc200/')
    base_dirs.append('/scratch3/xiaocanli/sigma1-mime100-beta001-mustang/')
    base_dirs.append('/scratch3/xiaocanli/mime25-guide0-beta001-200-100/')
    base_dirs.append('/scratch3/xiaocanli/mime25-guide0-beta001-200-100-sigma033/')
    base_dirs.append('/net/scratch2/xiaocanli/mime25-sigma1-beta002-200-100-noperturb/')
    run_names = []
    run_names.append('mime25_beta02')
    run_names.append('mime25_beta007')
    run_names.append('mime25_beta002')
    run_names.append('mime25_beta0007')
    run_names.append('mime100_beta002')
    run_names.append('mime25_beta002_sigma01')
    run_names.append('mime25_beta002_sigma033')
    run_names.append('mime25_beta002_noperturb')
    return (base_dirs, run_names)

if __name__ == "__main__":
    ApJ_long_paper_runs()
