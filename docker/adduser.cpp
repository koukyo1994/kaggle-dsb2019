#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include <grp.h>

#include <cstdlib>
#include <cstdio>

const char *getenv_default(const char *key, const char *default_value)
{
    const char *env = std::getenv(key);
    return env ? env : default_value;
}

int main(int argc, char *argv[], char *env[])
{
    int ret;

    const char *user = getenv_default("USER", "kaggle");
    const char *home = getenv_default("HOME", "/home/kaggle");
    const char *shell = getenv_default("SHELL", "/bin/bash");
    const uid_t uid = getuid();
    const uid_t gid = getgid();

    if (getpwuid(uid)) // existing user
        return 0;

    // Add user
    std::FILE *passwd_file = std::fopen("/etc/passwd", "a");
    if (!passwd_file)
    {
        std::perror("Couldn't open /etc/passwd");
        return 1;
    }
    struct passwd new_user
    {
        (char *)user, (char *)"*", uid, gid, (char *)user, (char *)home, (char *)shell
    };
    ret = putpwent(&new_user, passwd_file);
    fclose(passwd_file);
    if (ret == -1)
    {
        std::perror("Couldn't add user to /etc/passwd");
        return 1;
    }

    // Add group
    std::FILE *group_file = std::fopen("/etc/group", "a");
    if (!group_file)
    {
        std::perror("Couldn't open /etc/group");
        return 1;
    }
    struct group new_group
    {
        (char *)user, (char *)"*", gid, NULL
    };
    ret = putgrent(&new_group, group_file);
    fclose(group_file);
    if (ret == -1)
    {
        std::perror("Couldn't add group to /etc/group");
        return 1;
    }

    std::FILE *sudoers_file = std::fopen("/etc/sudoers.d/99-kaggle", "w");
    if (!sudoers_file)
    {
        std::perror("Couldn't open /etc/sudoers.d/99-kaggle");
        return 1;
    }
    fprintf(sudoers_file, "%s ALL = (ALL:ALL) NOPASSWD: ALL\n", user);
    fclose(sudoers_file);
}
