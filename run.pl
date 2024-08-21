#!/usr/bin/env perl
package MyApp;
use 5.020;
use warnings;
use strict;
use POSIX qw/strftime/;
use MooseX::App::Simple;

option 'width' => (
    is => 'rw',
    isa => 'Int',
    documentation => 'number of sites in y-direction',
    required => 1,
    );

option 'JJ-length' => (
is => 'rw',
    isa => 'Int',
    documentation => 'number of sites of junction in x-direction',
    required => 1
    );


option 'electrode-length' => (
is => 'rw',
    isa => 'Int',
    documentation => 'number of sites of electrodes in x-direction',
    default => 400,
    );


option 'disorder' => (
    is => 'rw',
    isa => 'Num',
    documentation => 'choose maximum disorder potential in  [-mu * disorder, mu * disorder]',
    default => 4,
    );

option 'disorder-points' => (
    is => 'rw',
    isa => 'Int',
    documentation => 'Number of disorder configurations between zero and the maximum disorder potential',
    default => 4,
    );

option 'mu' => (
    is => 'rw',
    isa => 'Num',
    documentation => 'chemical potential (meV)',
    default => 10,
    );



sub run {
    my ($self) = @_;

    # create working directory of process
    my $basename = sprintf("");
    $basename = strftime( '%H-%M-%S', localtime() ) . "_$basename";
    $basename = strftime( '%Y-%m-%d', localtime() ) . "_$basename";
    
    chdir $dir or die "Can't chdir to $dir: $!\n";
    
    my @command = ();
    system(@command);
}

use MyApp;
MyApp->new_with_options->run();




    
