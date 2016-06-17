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
        std::string dir = ".";
        const char * file_prefix = "frey";

        char strbuf [255];
        std::string msg;

        bool verbose = false;

        int n_in = 2000;
        int n_out = 10;

        double sparseness = 0.1;
        double weight = 0.05;
        double am = 2e-4;
        double ap = 5e-4;

        double noise = 1e-4;
        double tau = 200.;

        double pretime = 0.;
        double time = 0.25;
        double posttime = 3600.;

	AurynFloat monitor_time = 60.;

	string protocol = "wtet";
        bool dopamine = false;

        double zup = 0.33;

        int n_rec = 100;

        int errcode = 0;

        try {

                po::options_description desc("Allowed options");
                desc.add_options()
                        ("help,h", "produce help message")
                        ("verbose,v", "verbose mode")
                        ("pre", po::value<double>(), "pre time [0.]")
                        ("tot,t", po::value<double>(), "total time [3600.]")
                        ("sparseness,s", po::value<double>(), "sparseness [0.1]")
                        ("protocol,p", po::value<int>(), "protocol 0-3 [WTET,stet,wlfs,slfs]")
                        ("nrec,n", po::value<int>(), "number of recorded synapses [100]")
                        ("weight,w", po::value<double>(), "weight [0.05]")
                        ("am", po::value<double>(), "depression cte [2e-4]")
                        ("ap", po::value<double>(), "potentiation cte [5e-4]")
                        ("noise", po::value<double>(), "noise [1e-4]")
                        ("tau", po::value<double>(), "synaptic time cte [200]")
                        ("montime,m", po::value<double>(), "monitor record interval [60.]")
                        ("dir", po::value<string>(), "output dir")
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

                if (vm.count("pre")) {
                        std::cout << "pre time set to "
                                  << vm["pre"].as<double>() << ".\n";
                        pretime = vm["pre"].as<double>();
                }

                if (vm.count("tot")) {
                        std::cout << "total time set to "
                                  << vm["tot"].as<double>() << ".\n";
                        posttime = vm["tot"].as<double>();
                }

                if (vm.count("sparseness")) {
                        std::cout << "sparseness set to "
                                  << vm["sparseness"].as<double>() << ".\n";
                        sparseness = vm["sparseness"].as<double>();
                }

                if (vm.count("protocol")) {
                        int prot = vm["protocol"].as<int>();
                        std::cout << "protocol set to ";
                        switch (prot) {
                        case 0:
                                std::cout << "wtet.\n";
                                protocol = "wtet";
                                time = 0.25;
                                break;
                        case 1:
                                std::cout << "stet.\n";
                                protocol = "stet";
                                time = 1201.;
                                dopamine = true;
                                break;
                        case 2:
                                std::cout << "wlfs.\n";
                                protocol = "wlfs";
                                time = 900.;
                                break;
                        case 3:
                                std::cout << "slfs.\n";
                                protocol = "slfs";
                                time = 900.;
                                dopamine = true;
                                break;
                        }
                }

                if (vm.count("nrec")) {
                        n_rec = vm["nrec"].as<int>();
                        std::cout << "n_rec set to " << n_rec << ".\n";
                }

                if (vm.count("weight")) {
                        std::cout << "weight set to "
                                  << vm["weight"].as<double>() << ".\n";
                        weight = vm["weight"].as<double>();
                }

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

                if (vm.count("noise")) {
                        std::cout << "noise set to "
                                  << vm["noise"].as<double>() << ".\n";
                        noise = vm["noise"].as<double>();
                }

                if (vm.count("tau")) {
                        std::cout << "tau set to "
                                  << vm["tau"].as<double>() << ".\n";
                        tau = vm["tau"].as<double>();
                }

                if (vm.count("montime")) {
                        std::cout << "monitor_time set to "
                                  << vm["montime"].as<double>() << ".\n";
                        monitor_time = vm["montime"].as<double>();
                }

                if (vm.count("dir")) {
                        std::cout << "dir set to "
                                  << vm["dir"].as<string>() << ".\n";
                        dir = vm["dir"].as<string>();
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

        sprintf(strbuf, "./%s_%s.ras", file_prefix, protocol.c_str());
        FileInputGroup * tetanus = new FileInputGroup(n_in, strbuf);

        AIFGroup * neuron = new AIFGroup(n_out);

        msg =  "Setting up E connections ...";
        logger->msg(msg,PROGRESS,true);

        ZynapseConnection *con = \
                new ZynapseConnection(tetanus, neuron, weight, sparseness, am, ap);
        con->random_data_potentiation(zup);
        con->set_noise(noise);
        con->set_tau(tau,0);con->set_tau(tau,1);con->set_tau(tau,2);

        msg = "Setting up monitors ...";
        logger->msg(msg,PROGRESS,true);

	sprintf(strbuf, "%s/%s_%s_x.%d.wgs", dir.c_str(), file_prefix, protocol.c_str(), world.rank());
        WeightStatsMonitor * wsmon_x = \
                new WeightStatsMonitor(con, strbuf, monitor_time, 0);
	sprintf(strbuf, "%s/%s_%s_y.%d.wgs", dir.c_str(), file_prefix, protocol.c_str(), world.rank());
        WeightStatsMonitor * wsmon_y = \
                new WeightStatsMonitor(con, strbuf, monitor_time, 1);
	sprintf(strbuf, "%s/%s_%s_z.%d.wgs", dir.c_str(), file_prefix, protocol.c_str(), world.rank());
        WeightStatsMonitor * wsmon_z = \
                new WeightStatsMonitor(con, strbuf, monitor_time, 2);

	// to count states (directly on data files): if x_i>0 s+=2^i
	if (n_rec>0) {
		sprintf(strbuf, "%s/%s_%s_x.%d.syn", dir.c_str(), file_prefix, protocol.c_str(), world.rank());
		WeightMonitor * xmon =					\
			new WeightMonitor(con, 0, n_rec, strbuf, monitor_time, DATARANGE, 0);
		sprintf(strbuf, "%s/%s_%s_y.%d.syn", dir.c_str(), file_prefix, protocol.c_str(), world.rank());
		WeightMonitor * ymon =					\
			new WeightMonitor(con, 0, n_rec, strbuf, monitor_time, DATARANGE, 1);
		sprintf(strbuf, "%s/%s_%s_z.%d.syn", dir.c_str(), file_prefix, protocol.c_str(), world.rank());
		WeightMonitor * zmon =					\
			new WeightMonitor(con, 0, n_rec, strbuf, monitor_time, DATARANGE, 2);
	}

	sprintf(strbuf, "%s/%s_%s.%d.ras", dir.c_str(), file_prefix, protocol.c_str(), world.rank());
        SpikeMonitor * smon = new SpikeMonitor(neuron, strbuf);

	sprintf(strbuf, "%s/%s_%s.%d.zyn", dir.c_str(), file_prefix, protocol.c_str(), world.rank());
        ZynapseMonitor * zmon = new ZynapseMonitor(con, strbuf, monitor_time);

	msg = "Simulating ...";
	logger->msg(msg,PROGRESS,true);

        // pre
        if (!sys->run(pretime, false) )
                errcode = 1;
        // stimulus
        tetanus->active = true;
        if (!sys->run(time, false) )
                errcode = 1;
	if (dopamine) {
		neuron->dopamine_on();
		if (!sys->run(60, false) )
			errcode = 1;
		neuron->dopamine_off();
		posttime -= 60;
	}
	// post
	double remaining = posttime - time;
	if (!sys->run(remaining, false) )
		errcode = 1;

	msg = "Freeing ...";
        logger->msg(msg,PROGRESS,true);

        delete sys;

        if (errcode)
                env.abort(errcode);
        return errcode;
}
