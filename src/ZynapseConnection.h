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
#include "TripletConnection.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/exponential_distribution.hpp>
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
#define TAU_PRE 0.0168
#define TAU_POST 0.0337
#define TAU_LONG 0.04
#define C_RESET 1.

using namespace std;

/*! \brief Implements complex synapse as described by Ziegler et al. 2015.
 */
class ZynapseConnection : public TripletConnection
{
private:

	// TODO compare with LPT
        AurynFloat euler[3], eta;
        int t_updates;

	void init(AurynFloat wo, AurynFloat a_m, AurynFloat a_p,
		  AurynFloat k_w);

        static boost::mt19937 gen;
        boost::normal_distribution<> *dist;
        boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > * die;

        void free();

protected:

        /*! Action on weight upon presynaptic spike on connection with postsynaptic
         * partner post. This function should be modified to define new spike based
         * plasticity rules.
         * @param post the postsynaptic cell from which the synaptic trace is read out*/
        virtual void dw_pre(NeuronID * post, AurynWeight * weight);

        /*! Action on weight upon postsynaptic spike of cell post on connection
         * with presynaptic partner pre. This function should be modified to define
         * new spike based plasticity rules.
         * @param pre the presynaptic cell in question.
         * @param post the postsynaptic cell in question.
         */
        virtual void dw_post(NeuronID * pre, NeuronID post, AurynWeight * weight);

        void integrate();

        void noise(NeuronID z);
        void compute_diffs();

	// temporary state vector
	AurynWeight * temp_state;

        // AurynWeight xtow(AurynWeight value);

public:

        ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
			  TransmitterType transmitter);
        ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
			  AurynFloat wo, AurynFloat sparseness, TransmitterType transmitter);
	/*! Default constructor. Sets up a random sparse connection and plasticity parameters
	 *
	 * @param source the presynaptic neurons.
	 * @param destinatino the postsynaptic neurons.
	 * @param wo the initial synaptic weight and lower fixed point of weight dynamics.
	 * @param sparseness the sparseness of the connection (probability of connection).
	 * @param a_m the depression learning rate.
	 * @param a_p the potentiation learning rate.
	 * @param kw the relative high weight (default is 3).
	 * @param transmitter the TransmitterType (default is GLUT, glutamatergic).
	 * @param name a sensible identifier for the connection used in debug output.
	 */
        ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
			  AurynFloat wo, AurynFloat sparseness,
			  AurynFloat a_m, AurynFloat a_p, AurynFloat kw=KW,
			  TransmitterType transmitter=GLUT,
			  string name = "ZynapseConnection");
        ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
			  const char * filename, AurynFloat wo, AurynFloat a_m,
			  AurynFloat a_p, AurynFloat kw=KW,
			  TransmitterType transmitter=GLUT);

        virtual ~ZynapseConnection();

        virtual void propagate();
        virtual void evolve();

        void random_data_potentiation(AurynFloat z_up, bool reset=false);
        void count_states(AurynInt *states);

        void seed(int s);

        void set_plast_constants(AurynFloat a_m, AurynFloat a_p);
        void potentiate(NeuronID i);
        // void potentiate();
//         virtual void stats(AurynFloat &mean, AurynFloat &std);
//         void stats(AurynFloat &mean, AurynFloat &std, vector<NeuronID> * presynaptic_list);
};

#endif /*ZYNAPSECONNECTION_H_*/
