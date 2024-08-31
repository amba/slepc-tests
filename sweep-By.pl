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

has 'mu' => (
    is => 'rw',
    isa => 'Num',
    documentation => 'chemical potential (meV)',
    default => 10,
    );

has 'EZY' => (
    is => 'rw',
    isa => 'Num',
    documentation => 'zeeman field in y-direction',
    default => 0,
    );

has 'EZY_points' => (
    is => 'rw',
    isa => 'Int',
    documentation => 'Number of fields in y-direction',
    default => 11,
    );



has 'alpha' => (
    is => 'rw',
    isa => 'Num',
    documentation => 'Rashba SOC (in meV nm)',
    default => 0,
    );


has 'spectrum_range' => (
    is => 'rw',
    isa => 'Num',
    documentation => 'calculate spectrum up to N_ABS_bound_states * spectrum_range',
    default => 2,
    );


use MyApp;
my $cmd = '2d_spin';
my $app = MyApp->new_with_options();

system('make');


# create working directory of process
my $folder = sprintf(
    "EZY-SWEEP_width=%d_JJlength=%d_disorder=%g_alpha=%g",
    $app->width, $app->JJ_length, $app->disorder, $app->alpha,
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

my $EZY_max = $app->EZY;
my $EZY_step = $EZY_max / ($app->EZY_points - 1);
my @EZY_points = map {$_ * $EZY_step} (0..($app->EZY_points-1));
say "EZY values: @EZY_points";
for my $ezy (@EZY_points) {
    say "ezy = $ezy";
    my $output_filename = sprintf("output_EZY=%.7g.dat", $ezy);
    my @command = ("../$cmd",'-mu', $app->mu, '-JJlength', $app->JJ_length, '-leadlength', $app->lead_length, '-JJwidth', $app->width, '-dis', $app->disorder, '-EZX',0 , '-EZY', $ezy, '-alpha', $app->alpha, '-spectrum', $app->spectrum_range, '-output', $output_filename);
    say "running command: @command";
    
    # open my $cmd_fh, '>', 'cmd.yml' or die "cannot open $!";
    # print {$cmd_fh} Dump(\@command);

    system(@command);
}




    
