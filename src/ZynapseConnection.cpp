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

boost::mt19937 ZynapseConnection::gen = boost::mt19937();

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
	delete [] temp_state;
}

void ZynapseConnection::finalize() {
	// will compute backward matrix on the new elements/data vector of the w
	DuplexConnection::finalize();
}

void ZynapseConnection::init(AurynFloat wo, AurynFloat k_w, AurynFloat a_m, AurynFloat a_p)
{
        if (dst->get_post_size() == 0) return;

        dist = new boost::normal_distribution<> (0., 1.);
        die = new boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >
		(gen, *dist);
	seed(communicator->rank());

	set_min_weight(wo);
	set_max_weight(kw*wo);

	tr_pre->set_timeconstant(TAU_PRE);
	tr_post->set_timeconstant(TAU_POST);
	tr_post2->set_timeconstant(TAU_LONG);

        set_plast_constants(a_m, a_p);

        euler[0] = TUPD/TAUX;
        euler[1] = TUPD/TAUYZ;
        euler[2] = TUPD/TAUYZ;

        timestep_synapses = TUPD/dt;

        eta = sqrt(ETAXYZ*TUPD);

	// TODO write a Connection:fct to do all this automatically?
	// Set number of synaptic states
	w->set_num_synapse_states(3);

	// copy all the elements from z=0 to z=1,2
	w->state_set_all(w->get_state_begin(1),0.0);
	w->state_set_all(w->get_state_begin(2),0.0);
	w->state_add(w->get_state_begin(0),w->get_state_begin(1));
	w->state_add(w->get_state_begin(0),w->get_state_begin(2));

	/* Define temporary state vectors */
	// TODO what size?
	temp_state = new AurynWeight[w->get_statesize()];
	diff_state = new AurynWeight[2*w->get_statesize()];

	// Run finalize again to rebuild backward matrix
	finalize(); 
}

void ZynapseConnection::set_plast_constants(AurynFloat a_m, AurynFloat a_p)
{
        hom_fudge = a_m/TAU_POST;
        A3_plus = a_p/TAU_PRE/TAU_LONG;
}

/************
 *** body ***
 ************/

// TODO rewrite with new dynamics
void ZynapseConnection::integrate()
{
        compute_diffs();
        for (int z=0; z!=3; z++) {
                gsl_vector_float *x = layers[z];

                // X^3-X = X*(X*X-1)
                gsl_blas_scopy(x,bufs[0]);
                auryn_vector_float_mul(bufs[0],x);
                auryn_vector_float_add_constant(bufs[0],-1.);
                auryn_vector_float_mul(bufs[0],x);
                if (z==0) {
                        // -meta*(1-dxy)*(y-x)
                        gtogsl(bufs[1]);
                        auryn_vector_float_add_constant(bufs[1],-1.);
                        auryn_vector_float_mul(bufs[1],diffs[0]);
                        // auryn_vector_float_saxpy(-TILT,bufs[1],bufs[0]); // for noise tuning only !!
                        auryn_vector_float_saxpy(-META_YX,bufs[1],bufs[0]);
                }
                if (z==1) {
                        // -meta*(1-dyz)*(z-y)
                        auryn_vector_float_saxpy(META_ZY*(1.-dst->get_protein()),diffs[1],bufs[0]);
                        // -tilt*dxy*(x-y)
                        gtogsl(bufs[1]);
                        auryn_vector_float_mul(bufs[1],diffs[0]);
                        auryn_vector_float_saxpy(-TILT,bufs[1],bufs[0]);
                }
                if (z==2)
                        // -tilt*dyz*(y-z)
                        auryn_vector_float_saxpy(-TILT*dst->get_protein(),diffs[1],bufs[0]);
                // add (t-to)/tau*(-Xdot) to X
                auryn_vector_float_saxpy(-euler[z],bufs[0],x);
                // noise
                noise(z);
        }
        // diffs and couplings
	// TODO add proteins to neurongroup
        dst->update_prot();
}

void ZynapseConnection::noise(NeuronID z)
{
        AurynWeight *data_begin = w->get_data_begin(z);
        for (AurynWeight *dat=data_begin;
             dat!=(data_begin+w->get_statesize()); dat++)
                *dat += eta*(*die)();
}

// TODO difference statesize nonzero ?
void ZynapseConnection::compute_diffs()
{
	AurynWeight * x = w->get_state_begin(0),
		* y = w->get_state_begin(1),
		* z = w->get_state_begin(2),
		* dxy = diff_state,
		* dyz = diff_state+w->get_statesize();
	w->state_sub(x,y,dxy);
	w->state_sub(y,z,dyz);
}

// TODO adapt (careful with constants due to w instead of x)
// TODO comment
/*! This function implements what happens to synapes transmitting a
 *  spike to neuron 'post'. */
void ZynapseConnection::dw_pre(NeuronID * post, AurynWeight * weight)
{
        // translate post id to local id on rank: translated_spike
        NeuronID translated_spike = dst->global2rank(*post),
		data_ind = post-fwd_ind;
        AurynDouble dw = hom_fudge*tr_post->get(translated_spike),
                reset = gsl_vector_float_get(layers[2],data_ind)-*x; // TODO
        if (reset<0) dw *= 1-C_RESET*reset;
        if (dw>1) dw = 1;
        if (reset>0) tr_gxy->add(data_ind, dw*(1.-tr_gxy->get(data_ind))); // TODO add trace
        dw *= (1+*weight); // TODO whats this
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
                reset = gsl_vector_float_get(layers[2],data_ind)-*x;
        if (reset>0) dw *= 1+C_RESET*reset;
        if (dw>1) dw = 1;
        if (reset<0) tr_gxy->add(data_ind,dw*(1.-tr_gxy->get(data_ind)));
        dw *= (1-*weight);
        *weight += dw;
}

void ZynapseConnection::evolve()
{
        if (sys->get_clock()%timestep_synapses==0)
                integrate();
}

// AurynWeight ZynapseConnection::wtox(AurynWeight value)
// {
//         return (value*2/wo-(kw+1))/(kw-1);
// }

void ZynapseConnection::random_data_potentiation(AurynFloat z_up, bool reset)
{
        if (reset) {
		AurynWeight weight = get_min_weight();
		// TODO replace with set(z)
                set_weight(weight); set_tag(weight); set_scaffold(weight);
        }
        if (z_up) {
		// TODO check in sparseC
                boost::exponential_distribution<> exp_dist(z_up);
                boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > exp_die(gen, exp_dist);

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
#ifdef RANDOM_SEED
        timeval ss;
        gettimeofday(&ss,NULL);
        s += ss.tv_usec-100*(ss.tv_usec/100);
#endif
        gen.seed(s);
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

// TODO adapt to w
void ZynapseConnection::potentiate(NeuronID i)
{
  for (int z=0; z<3; z++)
    gsl_vector_float_set(layers[z],i,1.);
}

void ZynapseConnection::potentiate()
{
        for (int z=0; z<3; z++)
                gsl_vector_float_set_all(layers[z],1.);
}
