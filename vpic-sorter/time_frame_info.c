#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

/******************************************************************************
 * Get the total number of frames and time step interval.
 ******************************************************************************/
int get_time_frame_info(int *ntf, int *tinterval, char *path) {
    struct dirent* dent;
    DIR* srcdir = opendir(path);
    int tp1, tp2;

    tp1 = 0;
    *ntf = 0;
    if (srcdir == NULL)
    {
        perror("opendir");
        return -1;
    }

    while((dent = readdir(srcdir)) != NULL)
    {
        struct stat st;

        if(strcmp(dent->d_name, ".") == 0 || strcmp(dent->d_name, "..") == 0)
            continue;

        if (fstatat(dirfd(srcdir), dent->d_name, &st, 0) < 0)
        {
            perror(dent->d_name);
            continue;
        }

        if (S_ISDIR(st.st_mode)) {
            sscanf(dent->d_name, "T.%d", &tp2);
            *tinterval = tp2 - tp1;
            tp1 = tp2;
            (*ntf)++;
        }
    }
    closedir(srcdir);
    return 0;
}
