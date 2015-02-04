// TODO
// copyright?

/*
 * Copyright 2014 Friedemann Zenke
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
 *
 * If you are using Auryn or parts of it for your work please cite:
 * Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations
 * of spiking neural networks using general-purpose computers.
 * Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
 */

#ifndef ZYNAPSECONNECTION_H_
#define ZYNAPSECONNECTION_H_

#include "auryn_definitions.h"
#include "LPTripletConnection.h"

// TODO compare with Poisson f.ex.

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

/*! constants for the synapse model
 */
#define TILT 3.5*0.25
#define META_YX 1.3*0.25
#define META_ZY 0.95*0.25

#define TAUX 20.
#define TAUYZ 200.

#define TUPD 100e-3

#define TAUG 600.

#define ETAXYZ 0.0001

#define THETAG 0.37 // e^-1

#define KW 3 // 3 for frey

/*! constants for the plasticity rule
 */
#define AM 1e-3
#define AP 1e-3

using namespace std;

/*! \brief Implements complex synapse as described by Ziegler et al. 2015.
 */
class ZynapseConnection : public LPTripletConnection
{
private:

	// TODO what is it
        AurynFloat euler[3], eta;

        int t_updates;

	void init(AurynFloat w_o, AurynFloat a_m, AurynFloat a_p,
		  AurynFloat k_w, AurynFloat tau_hom, AurynFloat kappa);

        // TODO check in Poisson f.ex.
        static boost::mt19937 gen;
        static bool has_been_seeded;
        boost::normal_distribution<> *dist;
        boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > * die;

        void finalize(); // TODO need it?
        void free();

protected:

	// TODO need to rewrite?
	void propagate_forward();
        void propagate_backward();

        /*! Action on weight upon presynaptic spike on connection with postsynaptic
         * partner post. This function should be modified to define new spike based
         * plasticity rules.
         * @param post the postsynaptic cell from which the synaptic trace is read out*/
        // TODO x,i ?
	// TODO if not rewrite propagate -> make it virtual
        void dw_pre(AurynWeight *x, NeuronID i, NeuronID post);

        /*! Action on weight upon postsynaptic spike of cell post on connection
         * with presynaptic partner pre. This function should be modified to define
         * new spike based plasticity rules.
         * @param pre the presynaptic cell in question.
         * @param post the postsynaptic cell in question.
         */
        // TODO i ?
	// TODO if not rewrite propagate -> make it virtual
        void dw_post(NeuronID i, NeuronID pre, NeuronID post);

        void integrate();
        void noise(NeuronID z);
	// TODO put in integrate ?
        // void compute_diffs_loc(NeuronID i);
        // void compute_diffs(NeuronID z);
        // void compute_diffs();

        AurynWeight xtow(AurynWeight value);

public:

	// TODO see whats really needed + add a default
        ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
			  AurynFloat w_o, TransmitterType transmitter);
        ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
			  AurynFloat w_o, AurynFloat sparseness, TransmitterType transmitter);
        ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
			  AurynFloat w_o, AurynFloat sparseness,
			  AurynFloat tau_hom, AurynFloat kappa, TransmitterType transmitter,
			  string name = "ZynapseConnection");
        ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
			  AurynFloat w_o, AurynFloat sparseness,
			  AurynFloat a_m, AurynFloat a_mm, AurynFloat a_p, AurynFloat a_pp,
			  AurynFloat tau_hom, AurynFloat kappa, TransmitterType transmitter,
			  AurynFloat kw=KW, string name = "ZynapseConnection");
        ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
			  const char * filename, AurynFloat w_o, TransmitterType transmitter);
        ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
			  const char * filename, AurynFloat w_o, AurynFloat a_m, AurynFloat a_mm,
			  AurynFloat a_p, AurynFloat a_pp, AurynFloat tau_hom, AurynFloat kappa,
			  TransmitterType transmitter, AurynFloat kw=KW);

        virtual ~ZynapseConnection();

        virtual void propagate();
        virtual void evolve();

        void random_data_potentiation(AurynFloat z_up, bool reset=false);
        void count_states(AurynInt *states);

	// TODO check in Poisson f.ex.
        void seed(int s);

	// TODO see whats needed
//         void set_plast_constants(AurynFloat a_m, AurynFloat a_pp,
// 				 AurynFloat a_mm=0, AurynFloat a_p=0);
//         float *get_layer_ptr(int z);
//         /* NeuronID get_size(); */
//         void set_noise(AurynFloat level);
//         void set_tau(AurynInt z, AurynFloat level);
//         /* void set_x(NeuronID i, AurynFloat level); */
//         void set_x(AurynFloat level);
//         void set_y(AurynFloat level);
//         void set_z(AurynFloat level);
//         void potentiate_bkw(NeuronID j);
//         void potentiate_bkw_range(NeuronID n);
//         void potentiate(NeuronID i);
//         void potentiate();
//         void stabilize(NeuronID i);
//         void stabilize();
//         void fall(NeuronID i);
//         void fall();
//         void decay(NeuronID i);
//         void decay();
//         void rebounce(NeuronID i);
//         void rebounce();
//         void tag(NeuronID i);
//         void tag();
//         void depress();
//         AurynFloat get_g(NeuronID i);
//         void set_g(AurynFloat value);
//         virtual void stats(AurynFloat &mean, AurynFloat &std);
//         void stats(AurynFloat &mean, AurynFloat &std, vector<NeuronID> * presynaptic_list);
};

#endif /*ZYNAPSECONNECTION_H_*/
