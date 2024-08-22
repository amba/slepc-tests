#!/usr/bin/env perl
package MyApp;
use 5.020;
use warnings;
use strict;
use POSIX qw/strftime/;
use File::Copy 'copy';
use YAML::XS;
use Moose;
with 'MooseX::Getopt::Dashes';

has 'width' => (
    is => 'rw',
    isa => 'Int',
    documentation => 'number of sites in y-direction',
    required => 1,
    );

has 'JJ_length' => (
is => 'rw',
    isa => 'Int',
    documentation => 'number of sites of junction in x-direction',
    required => 1
    );


has 'lead_length' => (
is => 'rw',
    isa => 'Int',
    documentation => 'number of sites of electrodes in x-direction (default: 400)',
    default => 400,
    );


has 'disorder' => (
    is => 'rw',
    isa => 'Num',
    documentation => 'choose maximum disorder potential in  [-mu * disorder, mu * disorder] (default: 4)',
    default => 4,
    );

has 'disorder_points' => (
    is => 'rw',
    isa => 'Int',
    documentation => 'Number of disorder configurations between zero and the maximum disorder potential (default: 20)',
    default => 20,
    );

has 'mu' => (
    is => 'rw',
    isa => 'Num',
    documentation => 'chemical potential (meV)',
    default => 10,
    );




use MyApp;
my $cmd = '2d_no_spin';
my $app = MyApp->new_with_options();


# create working directory of process
my $folder = sprintf(
    "mu=%g_width=%d_JJlength=%d_leadlength=%d_disorder=%g",
    $app->mu, $app->width, $app->JJ_length, $app->lead_length,
    $app->disorder
    );

$folder = strftime( '%H-%M-%S', localtime() ) . "_$folder";
$folder = strftime( '%Y-%m-%d', localtime() ) . "_$folder";
say "folder: $folder";
mkdir $folder or die "cannot make folder '$folder': $!";
copy("${cmd}.c", "$folder/") or die "cannot copy: $!";
copy("run.pl" , "$folder/") or die "cannot copy: $!";

chdir $folder or die "Can't chdir to $folder: $!\n";

system('make');

# dump ARGV string
open my $argv_fh, '>', 'ARGV.yml' or die "cannot open $!";
print {$argv_fh} Dump($app->ARGV);

my @command = ('../2d_no_spin','-mu', $app->mu, '-JJlength', $app->JJ_length, '-leadlength', $app->lead_length, '-JJwidth', $app->width, '-dis', $app->disorder);
say "running command: @command";

open my $cmd_fh, '>', 'cmd.yml' or die "cannot open $!";
print {$cmd_fh} Dump(\@command);

system(@command);





    
