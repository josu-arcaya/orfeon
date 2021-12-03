#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import random
import re
import subprocess
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove

def line_in_file(file_path, pattern, subst):
    regexp = re.compile(pattern)
    found = False
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                if regexp.search(line):
                    found = True
                    new_file.write(subst)
                else:
                    new_file.write(line)
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)

    if not found:
        with open(file_path, 'a') as file:
            file.write(f"{subst}\n")

def apt_update():
    process = subprocess.Popen(['apt-get', 'update'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info("Finished update.")

def disable_dumps():
    line_in_file('/etc/security/limits.conf', '^\* hard core 0', '* hard core 0')
    logging.info("Finished disable_dumps.")

def minimum_password_age():
    line_in_file('/etc/login.defs', '^PASS_MIN_DAYS\s+\d', 'PASS_MIN_DAYS   1')
    logging.info("Finished minimum_password_age.")

def maximum_password_age():
    line_in_file('/etc/login.defs', '^PASS_MAX_DAYS\s+\d', 'PASS_MAX_DAYS   90')
    logging.info("Finished maximum_password_age.")

def default_umask():
    line_in_file('/etc/login.defs', '^UMASK\s+\d+', 'UMASK           027')
    logging.info("Finished default_umask.")

"""
https://cisofy.com/lynis/controls/PKGS-7370/
"""
def install_debsums():
    apt_update()
    process = subprocess.Popen(['apt-get', 'install', 'debsums', '-y'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info("Finished install_debsums.")

"""
https://cisofy.com/lynis/controls/PKGS-7394/
"""
def install_aptshowversions():
    apt_update()
    process = subprocess.Popen(['apt-get', 'install', 'apt-show-versions', '-y'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info("Finished install_aptshowversions.")

def ssh_config():
    line_in_file('/etc/ssh/sshd_config', '^AllowTcpForwarding', 'AllowTcpForwarding no')
    line_in_file('/etc/ssh/sshd_config', '^ClientAliveCountMax', 'ClientAliveCountMax 2')
    line_in_file('/etc/ssh/sshd_config', '^Compression', 'Compression no')
    line_in_file('/etc/ssh/sshd_config', '^LogLevel', 'LogLevel VERBOSE')
    line_in_file('/etc/ssh/sshd_config', '^MaxAuthTries', 'MaxAuthTries 3')
    line_in_file('/etc/ssh/sshd_config', '^MaxSessions', 'MaxSessions 2')
    line_in_file('/etc/ssh/sshd_config', '^TCPKeepAlive', 'TCPKeepAlive no')
    line_in_file('/etc/ssh/sshd_config', '^X11Forwarding', 'X11Forwarding no')
    line_in_file('/etc/ssh/sshd_config', '^AllowAgentForwarding', 'AllowAgentForwarding no')
    logging.info("Finished ssh_config.")

def legal_banner():
    line_in_file('/etc/ssh/sshd_config', '^Banner', 'Banner /etc/issue.net')
    logging.info("Finished legal_banner.")

"""
https://cisofy.com/lynis/controls/PKGS-9626/
"""
def install_sysstat():
    apt_update()
    process = subprocess.Popen(['apt-get', 'install', 'sysstat', '-y'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info("Finished install_sysstat.")

"""
https://cisofy.com/lynis/controls/PKGS-9628/
"""
def install_auditd():
    apt_update()
    process = subprocess.Popen(['apt-get', 'install', 'auditd', '-y'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info("Finished install_auditd.")

"""
https://cisofy.com/lynis/controls/FINT-4350/
"""
def install_aide():
    apt_update()
    process = subprocess.Popen(['apt-get', 'install', 'aide', '-y'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info("Finished install_aide.")

"""
https://cisofy.com/lynis/controls/HRDN-7230/
"""
def install_rkhunter():
    apt_update()
    process = subprocess.Popen(['apt-get', 'install', 'rkhunter', '-y'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info("Finished install_rkhunter.")

"""
https://cisofy.com/lynis/controls/KRNL-6000/
"""
def tune_sysctl_1():
    line_in_file('/etc/sysctl.conf', '^dev.tty.ldisc_autoload', 'dev.tty.ldisc_autoload=0')
    line_in_file('/etc/sysctl.conf', '^fs.protected_fifos', 'fs.protected_fifos=2')
    line_in_file('/etc/sysctl.conf', '^fs.suid_dumpable', 'fs.suid_dumpable=0')
    line_in_file('/etc/sysctl.conf', '^kernel.core_uses_pid', 'kernel.core_uses_pid=1')
    line_in_file('/etc/sysctl.conf', '^kernel.dmesg_restrict', 'kernel.dmesg_restrict=1')
    line_in_file('/etc/sysctl.conf', '^kernel.kptr_restrict', 'kernel.kptr_restrict=2')

def tune_sysctl_2():
    line_in_file('/etc/sysctl.conf', '^kernel.modules_disabled', 'kernel.modules_disabled=1')
    line_in_file('/etc/sysctl.conf', '^kernel.sysrq', 'kernel.sysrq=0')
    line_in_file('/etc/sysctl.conf', '^kernel.unprivileged_bpf_disabled', 'kernel.unprivileged_bpf_disabled=1')
    line_in_file('/etc/sysctl.conf', '^net.core.bpf_jit_harden', 'net.core.bpf_jit_harden=2')
    line_in_file('/etc/sysctl.conf', '^net.ipv4.conf.all.accept_redirects', 'net.ipv4.conf.all.accept_redirects=0')
    line_in_file('/etc/sysctl.conf', '^net.ipv4.conf.all.log_martians', 'net.ipv4.conf.all.log_martians=1')

def tune_sysctl_3():
    line_in_file('/etc/sysctl.conf', '^net.ipv4.conf.all.rp_filter', 'net.ipv4.conf.all.rp_filter=1')
    line_in_file('/etc/sysctl.conf', '^net.ipv4.conf.all.send_redirects', 'net.ipv4.conf.all.send_redirects=0')
    line_in_file('/etc/sysctl.conf', '^net.ipv4.conf.default.accept_redirects', 'net.ipv4.conf.default.accept_redirects=0')
    line_in_file('/etc/sysctl.conf', '^net.ipv4.conf.default.accept_source_route', 'net.ipv4.conf.default.accept_source_route=0')
    line_in_file('/etc/sysctl.conf', '^net.ipv4.conf.default.log_martians', 'net.ipv4.conf.default.log_martians=1')
    line_in_file('/etc/sysctl.conf', '^net.ipv6.conf.all.accept_redirects', 'net.ipv6.conf.all.accept_redirects=0')
    line_in_file('/etc/sysctl.conf', '^net.ipv6.conf.default.accept_redirects', 'net.ipv6.conf.default.accept_redirects=0')

    process = subprocess.Popen(['sysctl', '-p'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info("Finished tune_sysctl.")

def upgrade():
    apt_update()
    
    process = subprocess.Popen(['apt-get', 'upgrade', '-y'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info("Finished upgrade.")

def harden_it():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s')

    f = []
    f.append(disable_dumps)
    f.append(minimum_password_age)
    f.append(maximum_password_age)
    f.append(default_umask)
    f.append(install_debsums)
    f.append(install_aptshowversions)
    f.append(ssh_config)
    f.append(legal_banner)
    f.append(install_sysstat)
    f.append(install_auditd)
    # to enable aide need to configure unattended postfix install
    # f.append(install_aide)
    f.append(tune_sysctl_1)
    f.append(tune_sysctl_2)
    f.append(tune_sysctl_3)
    f.append(upgrade)

    number_of_functions = len(f)
    t = random.randint(0, number_of_functions)
    mask = [True]*t + [False]*(number_of_functions-t)
    random.shuffle(mask)
    logging.info(f"The mask is {mask}.")

    for i, m in enumerate(mask):
        logging.info(f"Function {i} has a {m} mask.")
        if m:
            f[i]()

if __name__ == '__main__':
    harden_it()
