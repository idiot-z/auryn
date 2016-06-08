// TODO copyright?
/*
 * Copyright 2014-2016 Friedemann Zenke
 *
 * This file is part of Auryn, a simulation package for plastic
 * spiking neural networks.
 *
 * Auryn is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Auryn is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Auryn.  If not, see <http://www.gnu.org/licenses/>.
 */


#include "auryn.h"

namespace po = boost::program_options;
namespace mpi = boost::mpi;

using namespace auryn;

int main(int ac, char* av[])
{

        std::string dir = "/home/ziegler/git/auryn/data/";
        const char * file_prefix = "zyn";

        char strbuf [255];
        std::string msg;

	bool verbose = false;
	bool fast = false;

        NeuronID size_in = 500;
        NeuronID size_out = 1000;
        double rate_e = 5;
        // double rate_i = 5;

        double am = 1e-4;
        double ap = 1e-4;
        double sparseness = 0.1;
        double zup = 0.1;
        AurynWeight we = 0.15;
        // AurynWeight wi = 0.3;

        double pretime = 10.;
        double simtime = 10.;

        int errcode = 0;

        try {

                po::options_description desc("Allowed options");
                desc.add_options()
                        ("help,h", "produce help message")
			("verbose,v", "verbose mode")
			("fast,f", "turn off some of the monitors to run faster")
                        ("dir", po::value<std::string>(), "output dir")
                        ("re", po::value<double>(), "excitatory input rate")
                        // ("ri", po::value<double>(), "inhibitory input rate")
                        ("we", po::value<double>(), "input weight (exc)")
                        ("zup,z", po::value<double>(), "initial potentiated synapses")
                        // ("wi", po::value<double>(), "input weight (inh)")
                        ("time,t", po::value<double>(), "simulation time")
                        ("am", po::value<double>(), "depression rate")
                        ("ap", po::value<double>(), "potentiation rate")
                        ("nin", po::value<int>(), "poisson size")
                        ("nout", po::value<int>(), "neuron size")
                        ;

                po::variables_map vm;
                po::store(po::parse_command_line(ac, av, desc), vm);
                po::notify(vm);

                if (vm.count("help")) {
                        std::cout << desc << "\n";
                        return 1;
                }

		if (vm.count("verbose")) {
			verbose = true;
		}

		if (vm.count("fast")) {
			std::cout << "fast on.\n";
			fast = true;
		}

                if (vm.count("dir")) {
                        std::cout << "dir set to "
                                  << vm["dir"].as<std::string>() << ".\n";
                        dir = vm["dir"].as<std::string>();
                }

                if (vm.count("re")) {
                        std::cout << "rate_e set to "
                                  << vm["re"].as<double>() << ".\n";
                        rate_e = vm["re"].as<double>();
                }

                // if (vm.count("ri")) {
                //         std::cout << "rate_i set to "
                //                   << vm["ri"].as<double>() << ".\n";
                //         rate_i = vm["ri"].as<double>();
                // }

                if (vm.count("time")) {
                        std::cout << "simtime set to "
                                  << vm["time"].as<double>() << ".\n";
                        simtime = vm["time"].as<double>();
                }

                if (vm.count("we")) {
                        std::cout << "we set to "
                                  << vm["we"].as<double>() << ".\n";
                        we = vm["we"].as<double>();
                }

                if (vm.count("zup")) {
                        std::cout << "zup set to "
                                  << vm["zup"].as<double>() << ".\n";
                        zup = vm["zup"].as<double>();
                }

                // if (vm.count("wi")) {
                //         std::cout << "wi set to "
                //                   << vm["wi"].as<double>() << ".\n";
                //         wi = vm["wi"].as<double>();
                // }

                if (vm.count("am")) {
                        std::cout << "am set to "
                                  << vm["am"].as<double>() << ".\n";
                        am = vm["am"].as<double>();
                }

                if (vm.count("ap")) {
                        std::cout << "ap set to "
                                  << vm["ap"].as<double>() << ".\n";
                        ap = vm["ap"].as<double>();
                }

                if (vm.count("nin")) {
                        std::cout << "size_in set to "
                                  << vm["nin"].as<int>() << ".\n";
                        size_in = vm["nin"].as<int>();
                }

                if (vm.count("nout")) {
                        std::cout << "size_out set to "
                                  << vm["nout"].as<int>() << ".\n";
                        size_out = vm["nout"].as<int>();
                }

        }
        catch(std::exception& e) {
		std::cerr << "error: " << e.what() << "\n";
                return 1;
        }
        catch(...) {
		std::cerr << "Exception of unknown type!\n";
        }

        // BEGIN Global stuff
        mpi::environment env(ac, av);
        mpi::communicator world;
        communicator = &world;

        sprintf(strbuf, "%s/%s.%d.log", dir.c_str(), file_prefix, world.rank());
        std::string logfile = strbuf;

	LogMessageType log_level_file = PROGRESS;
	if ( verbose ) log_level_file = EVERYTHING;
	logger = new Logger(logfile,world.rank(),PROGRESS,log_level_file);

        sys = new System(&world);
        // END Global stuff

	msg =  "Setting up neuron groups ...";
	logger->msg(msg,PROGRESS,true);

        PoissonGroup * poisson_e = new PoissonGroup(size_in, rate_e);
        // PoissonGroup * poisson_i = new PoissonGroup(size/4, rate_i);

        IFGroup * neurons = new IFGroup(size_out);

	// initialize membranes
	neurons->set_tau_mem(10e-3);
	neurons->random_mem(-60e-3,10e-3);

	msg =  "Setting up E connections ...";
	logger->msg(msg,PROGRESS,true);

        ZynapseConnection * con_e = new ZynapseConnection(poisson_e,neurons,we,sparseness,am,ap);
	con_e->random_data_potentiation(zup);
        // SparseConnection * con_e = new SparseConnection(poisson_e,neurons,we,sparseness,GLUT);

	msg = "Setting up monitors ...";
	logger->msg(msg,PROGRESS,true);

        sprintf(strbuf, "%s/%s.%d.ras", dir.c_str(), file_prefix, world.rank() );
        SpikeMonitor * smon_e = new SpikeMonitor( neurons, strbuf, 100);

        // sprintf(strbuf, "%s/%s.%d.mem", dir.c_str(), file_prefix.c_str(), world.rank() );
        // VoltageMonitor * vmon = new VoltageMonitor( neurons, 0, strbuf, 10 );

        // sprintf(strbuf, "%s/%s.%d.ampa", dir.c_str(), file_prefix.c_str(), world.rank() );
        // AmpaMonitor * ampamon = new AmpaMonitor( neurons, 0, strbuf, 10 );

        // sprintf(strbuf, "%s/%s.%d.gaba", dir.c_str(), file_prefix.c_str(), world.rank() );
        // GabaMonitor * gabamon = new GabaMonitor( neurons, 0, strbuf, 10 );

        sprintf(strbuf, "%s/%s.%d.prate", dir.c_str(), file_prefix, world.rank() );
        PopulationRateMonitor * pmon = new PopulationRateMonitor( neurons, strbuf, 1. );

	if (!fast) {
		sprintf(strbuf, "%s/%s.%d.syn", dir.c_str(), file_prefix, world.rank() );
		WeightMonitor * wmon = new WeightMonitor( con_e, strbuf, 10 );
		for ( int i = 0 ; i < size_in ; ++i )
			wmon->add_to_list(i,0);
	}

        // sprintf(strbuf, "%s/%s.%d.wi", dir.c_str(), file_prefix.c_str(), world.rank() );
        // WeightSumMonitor * wmoni = new WeightSumMonitor( con_i, strbuf );

        // RateChecker * chk = new RateChecker( neurons , -1 , 20. , 10);

        con_e->stdp_active = false;

	msg = "Simulating (plasticity off) ...";
	logger->msg(msg,PROGRESS,true);
        if (!sys->run(pretime,false))
                errcode = 1;

        con_e->stdp_active = true;

	msg = "Simulating (plasticity on) ...";
	logger->msg(msg,PROGRESS,true);
        if (!sys->run(simtime,false))
                errcode = 1;

	if (!fast) {
		msg = "Saving weight matrix ...";
		logger->msg(msg,PROGRESS,true);
		sprintf(strbuf, "%s/%s.%d.wmat", dir.c_str(), file_prefix, world.rank() );
		if (!con_e->write_to_file(strbuf))
			errcode = 1;

		msg = "Reloading weight matrix ...";
		logger->msg(msg,PROGRESS,true);
		if (!con_e->load_from_file(strbuf))
			errcode = 1;

		msg = "Simulating after reload ...";
		logger->msg(msg,PROGRESS,true);
		if (!sys->run(pretime,false))
			errcode = 1;
	}

	msg = "Freeing ...";
        logger->msg(msg,PROGRESS,true);
        delete sys;

        if (errcode)
                env.abort(errcode);
        return errcode;
}
