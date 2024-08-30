#!/usr/bin/env perl
package MyApp;
use 5.020;
use warnings;
use strict;
use POSIX qw/strftime/;
use File::Copy 'copy';
use YAML::XS;
use Moose;
use autodie qw/system/;

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
    default => 0,
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

has 'pairing_density' => (
    is => 'rw',
    isa => 'Num',
    documentation => 'pairing density',
    default => 1,
    );

# has 'EZX' => (
#     is => 'rw',
#     isa => 'Num',
#     documentation => 'zeeman field in x-direction',
#     default => 0,
#     );

has 'EZY' => (
    is => 'rw',
    isa => 'Num',
    documentation => 'zeeman field in y-direction',
    default => 0,
    );



has 'alpha' => (
    is => 'rw',
    isa => 'Num',
    documentation => 'Rashba SOC (in meV nm)',
    default => 0,
    );



use MyApp;
my $cmd = '2d_spin';
my $app = MyApp->new_with_options();

system('make');


# create working directory of process
my $folder = sprintf(
    "Disorder-sweep_width=%d_JJlength=%d_EZY=%g_alpha=%g",
    $app->width, $app->JJ_length, $app->EZY, $app->alpha,
    );

$folder = strftime( '%H-%M-%S', localtime() ) . "_$folder";
$folder = strftime( '%Y-%m-%d', localtime() ) . "_$folder";
say "folder: $folder";
mkdir $folder or die "cannot make folder '$folder': $!";
copy("${cmd}.c", "$folder/") or die "cannot copy: $!";
say "script name = $0";
copy($0 , "$folder/") or die "cannot copy: $!";

chdir $folder or die "Can't chdir to $folder: $!\n";


# dump ARGV string
open my $argv_fh, '>', 'ARGV.yml' or die "cannot open $!";
print {$argv_fh} Dump($app->ARGV);

my $max_disorder = $app->disorder;
my $disorder_step = $max_disorder / ($app->disorder_points - 1);
my @disorder_points = map {$_ * $disorder_step} (0..($app->disorder_points-1));
say "disorder values: @disorder_points";
for my $disorder (@disorder_points) {
    say "disorder = $disorder";
    my $output_filename = sprintf("output_disorder=%.7g.dat", $disorder);
    my @command = ("../$cmd",'-mu', $app->mu, '-JJlength', $app->JJ_length, '-leadlength', $app->lead_length, '-JJwidth', $app->width, '-dis', $disorder, '-EZX',0 , '-EZY', $app->EZY, '-alpha', $app->alpha, '-pairing_density', $app->pairing_density, '-output', $output_filename);
    say "running command: @command";
    
    # open my $cmd_fh, '>', 'cmd.yml' or die "cannot open $!";
    # print {$cmd_fh} Dump(\@command);

    system(@command);
}




    
