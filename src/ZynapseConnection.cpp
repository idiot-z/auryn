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

#include "ZynapseConnection.h"

// TODO add loggers, copyright?

using namespace auryn;

boost::mt19937 ZynapseConnection::zynapse_connection_gen = boost::mt19937();
bool ZynapseConnection::has_been_seeded = false;

/********************
 *** constructors ***
 ********************/

ZynapseConnection::ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
                                     TransmitterType transmitter)
        : TripletConnection(source, destination, transmitter)

{
        init(1, KW, AM, AP);
}

ZynapseConnection::ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
				     AurynFloat wo, AurynFloat sparseness,
				     TransmitterType transmitter)
        : TripletConnection(source, destination, wo, sparseness, 0, 1, 1, 1,
			    transmitter, "ZynapseConnection")

{
        init(wo, KW, AM, AP);
}

ZynapseConnection::ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
				     AurynFloat wo, AurynFloat sparseness,
				     AurynFloat a_m, AurynFloat a_p, AurynFloat kw,
				     TransmitterType transmitter, string name)
        : TripletConnection(source, destination, wo, sparseness, 0, 1, 1, 1,
			    transmitter, name)

{
        init(wo, kw, a_m, a_p);
        if ( name.empty() )
                set_name("ZynapseConnection");
}

ZynapseConnection::ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
				     const char *filename, AurynFloat wo,
				     AurynFloat a_m, AurynFloat a_p, AurynFloat kw,
				     TransmitterType transmitter)
        : TripletConnection(source, destination, filename, 0, 1, 1, 1, transmitter)
{
        init(wo, kw, a_m, a_p);
}

/*****************
 *** init crap ***
 *****************/

ZynapseConnection::~ZynapseConnection()
{
        if (dst->get_post_size() > 0)
                free();
}

void ZynapseConnection::free()
{
        delete dist;
        delete die;
	delete tr_gxy;
}

void ZynapseConnection::init(AurynFloat wo, AurynFloat k_w, AurynFloat a_m, AurynFloat a_p)
{
        if (dst->get_post_size() == 0) return;

        dist = new boost::normal_distribution<> (0., 1.);
        die = new boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >
		(zynapse_connection_gen, *dist);
	if (!has_been_seeded)
		seed(12345*auryn::communicator->rank());

	set_min_weight(wo);
	set_max_weight(k_w*wo);

        tr_pre = src->get_pre_trace(TAU_PRE);
        tr_post = dst->get_post_trace(TAU_POST);
        tr_post2 = dst->get_post_trace(TAU_LONG);

        set_plast_constants(a_m, a_p);

        euler[0] = TUPD/TAUX;
        euler[1] = TUPD/TAUY;
        euler[2] = TUPD/TAUZ;

        coeff[0] = 4/wo/wo/(k_w-1)/(k_w-1);
        coeff[1] = wo*wo*wo*k_w*(1+k_w)/2;
        coeff[2] = wo*wo*(1+k_w)*(1+k_w)/2 + wo*wo*k_w;
        coeff[3] = 3*wo*(1+k_w)/2;

        timestep_synapses = TUPD/dt;

        eta = wo*(k_w-1)*sqrt(ETAXYZ*TUPD)/2;

	// Set number of synaptic states
	w->set_num_synapse_states(3);

	// copy all the elements from z=0 to z=1,2
	w->state_set_all(w->get_state_begin(1),0.0);
	w->state_set_all(w->get_state_begin(2),0.0);
	w->state_add(w->get_state_begin(0),w->get_state_begin(1));
	w->state_add(w->get_state_begin(0),w->get_state_begin(2));

	// Run finalize again to rebuild backward matrix
	finalize(); 
}

void ZynapseConnection::set_plast_constants(AurynFloat a_m, AurynFloat a_p)
{
        hom_fudge = a_m/TAU_POST;
        A3_plus = a_p/TAU_PRE/TAU_LONG;
}

void ZynapseConnection::finalize() {
        TripletConnection::finalize();
	tr_gxy = new LinearTrace(get_nonzero(), TAUG);
}

/************
 *** body ***
 ************/

void ZynapseConnection::integrate()
{
	AurynWeight *x = w->get_state_begin(0),
		*y = w->get_state_begin(1),
		*z = w->get_state_begin(2);
	
	for (AurynLong i = 0 ; i < w->get_nonzero() ; ++i ) {
		AurynWeight xyi = x[i] - y[i],
			yzi = y[i] - z[i],
			gxy = tr_gxy->get(i);
		x[i] += euler[0]*(coeff[0]*(coeff[1]-x[i]*(coeff[2]-x[i]*(coeff[3]-x[i]) ) ) -
				  META_YX*(1-gxy)*xyi
				  ) + eta*(*die)();
		y[i] += euler[1]*(coeff[0]*(coeff[1]-y[i]*(coeff[2]-y[i]*(coeff[3]-y[i]) ) ) +
				  TILT*gxy*xyi -
				  META_ZY*(1-dst->get_protein())*yzi
				  ) + eta*(*die)();
		z[i] += euler[2]*(coeff[0]*(coeff[1]-z[i]*(coeff[2]-z[1]*(coeff[3]-z[i]) ) ) +
				  TILT*dst->get_protein()*yzi
				  ) + eta*(*die)();				  
	}
        dst->update_protein();
}

// TODO comment
/*! This function implements what happens to synapes transmitting a
 *  spike to neuron 'post'. */
void ZynapseConnection::dw_pre(NeuronID * post, AurynWeight * weight)
{
        // translate post id to local id on rank: translated_spike
        NeuronID translated_spike = dst->global2rank(*post),
		data_ind = post-fwd_ind;
	// NOTE get_data(data_ind) = get_data(post) !
        AurynDouble dw = hom_fudge*tr_post->get(translated_spike),
                reset = w->get_data(post,2)-*weight;
        if (reset<0) dw *= 1-C_RESET*reset;
        if (dw>1) dw = 1;
        if (reset>0) tr_gxy->add(data_ind, dw*(1.-tr_gxy->get(data_ind)));
        dw *= (*weight-wmin);
        *weight -= dw;
}

/*! This function implements what happens to synapes experiencing a
 *  backpropagating action potential from neuron 'pre'. */
void ZynapseConnection::dw_post(NeuronID * pre, NeuronID post, AurynWeight * weight)
{
        // at this point post was already translated to a local id in
        // the propagate_backward function below.
	NeuronID data_ind = bkw_data[pre-bkw_ind]-fwd_data;
        AurynDouble dw = A3_plus*tr_pre->get(*pre)*tr_post2->get(post),
                reset = w->get_data(data_ind,2)-*weight;
        if (reset>0) dw *= 1+C_RESET*reset;
        if (dw>1) dw = 1;
        if (reset<0) tr_gxy->add(data_ind,dw*(1.-tr_gxy->get(data_ind)));
        dw *= (wmax-*weight);
        *weight += dw;
}

void ZynapseConnection::evolve()
{
        if (sys->get_clock()%timestep_synapses==0)
                integrate();
}

void ZynapseConnection::random_data_potentiation(AurynFloat z_up, bool reset)
{
        if (reset) {
                depress();
        }
        if (z_up) {
                boost::exponential_distribution<> exp_dist(z_up);
                boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> >
			exp_die(zynapse_connection_gen, exp_dist);

                AurynLong x = (AurynLong) exp_die(), y;
                while (x < get_nonzero() ) {
                        potentiate(x);
                        y = (AurynLong)(exp_die()+0.5);
                        x += (y==0)?1:y;
                }
        }
}

// void ZynapseConnection::count_states(AurynInt *states)
// {
//         float *data[3];
//         for (int j=0; j<3; j++)
//                 data[j] = layers[j]->data;
//         for (NeuronID i=0; i<get_nonzero(); i++,data[0]++,data[1]++,data[2]++) {
//                 int state = 0;
//                 for (int j=0; j<3; j++)
//                         if ((*data[j])>0)
//                                 state += pow(2,j);
//                 states[state]++;
//         }
// }

// TODO check where is RANDOM_SEED + gettimeofday
void ZynapseConnection::seed(int s)
{
// #ifdef RANDOM_SEED
//         timeval ss;
//         gettimeofday(&ss,NULL);
//         s += ss.tv_usec-100*(ss.tv_usec/100);
// #endif
	std::stringstream oss;
	oss << get_log_name() << "Seeding with " << s;
	auryn::logger->msg(oss.str(),VERBOSE);
	zynapse_connection_gen.seed(s); 
	has_been_seeded = true;
}

// TODO still exists?
// void ZynapseConnection::stats(AurynFloat &mean, AurynFloat &std)
// {
//         NeuronID count = get_nonzero();
//         AurynFloat sum = 0;
//         AurynFloat sum2 = 0;
//         float *x = gsl_data;
//         for ( NeuronID i = 0 ; i < count ; ++i,++x ) {
//                 sum += *x;
//                 sum2 += pow(*x,2);
//         }
//         if ( count <= 1 ) {
//                 mean = sum;
//                 std = 0;
//                 return;
//         }
//         mean = sum/count;
//         std = sqrt(sum2/count-pow(mean,2));
// }

// void ZynapseConnection::stats(AurynFloat &mean, AurynFloat &std, vector<NeuronID> * presynaptic_list)
// {
//         NeuronID count = 0;
//         AurynFloat sum = 0;
//         AurynFloat sum2 = 0;
//         vector<NeuronID>::const_iterator i_end = presynaptic_list->end();
//         NeuronID * ind = w->get_row_begin(0);
//         for ( vector<NeuronID>::const_iterator i = presynaptic_list->begin(); i != i_end; ++i)
//                 for ( NeuronID * c = w->get_row_begin(*i); c != w->get_row_end(*i); ++c) {
//                         AurynWeight value = gsl_data[c-ind];
//                         sum += value;
//                         sum2 += pow(value,2);
//                         ++count;
//                 }
//         if ( count <= 1 ) {
//                 mean = sum;
//                 std = 0;
//                 return;
//         }
//         mean = sum/count;
//         std = sqrt(sum2/count-pow(mean,2));
// }

void ZynapseConnection::potentiate(NeuronID i)
{
  for (int z=0; z<3; z++)
	  w->state_data(i+z*w->get_nonzero(),get_max_weight());
}

void ZynapseConnection::potentiate()
{
	aurynWeight wmax = get_max_weight();
        for (int z=0; z<3; z++) {
		w->state_set_all(w->get_state_begin(z),wmax);
	}
}

void ZynapseConnection::depress()
{
	aurynWeight wmin = get_min_weight();
        for (int z=0; z<3; z++) {
		w->state_set_all(w->get_state_begin(z),wmin);
	}
}
