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

// TODO check in Poisson f.ex.
// static members
boost::mt19937 ZynapseConnection::gen = boost::mt19937();
bool ZynapseConnection::has_been_seeded = false;

/********************
 *** constructors ***
 ********************/

// nada
ZynapseConnection::ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
                                     TransmitterType transmitter)
        : TripletConnection(source, destination, transmitter)

{
	// TODO default w_0
        init(1, AM, AP, KW);
}

// sparseness
ZynapseConnection::ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
				     AurynFloat w_o, AurynFloat sparseness,
				     TransmitterType transmitter)
        : TripletConnection(source, destination, w_o, sparseness, 0, 1, 1, 1, transmitter)

{
        init(w_o, KW, AM, AP);
	// HERE continue with constructors & alloc_vectors ?
        alloc_vectors();
        init_shortcuts();
}

// sparseness, homeostasis
ZynapseConnection::ZynapseConnection(SpikingGroup *source, NeuronGroup *destination, AurynFloat w_o, AurynFloat sparseness, AurynFloat tau_hom, AurynFloat kappa, TransmitterType transmitter, string name)
        : TripletConnection(source, destination, w_o, sparseness, transmitter, name)

{
        init(w_o, KW, tau_hom, kappa, A2M, A3M, A2P, A3P);
        alloc_vectors();
        init_shortcuts();
}

// sparseness, plasticity, homeostasis
ZynapseConnection::ZynapseConnection(SpikingGroup *source, NeuronGroup *destination, AurynFloat w_o, AurynFloat sparseness, AurynFloat a_m, AurynFloat a_mm, AurynFloat a_p, AurynFloat a_pp, AurynFloat tau_hom, AurynFloat kappa, TransmitterType transmitter, AurynFloat kw, string name)
        : TripletConnection(source, destination, w_o, sparseness, transmitter, name)

{
        init(w_o, kw, tau_hom, kappa, a_m, a_mm, a_p, a_pp);
        alloc_vectors();
        init_shortcuts();
}

// filename
ZynapseConnection::ZynapseConnection(SpikingGroup *source, NeuronGroup *destination, const char *filename, AurynFloat w_o, TransmitterType transmitter)
        : TripletConnection(source, destination, transmitter)
{
        init(w_o, KW, TAU_HOM, KAPPA, A2M, A3M, A2P, A3P);
        if (! load_from_file(filename) )
                throw AurynMMFileException();
}

// filename, plasticity, homeostasis
ZynapseConnection::ZynapseConnection(SpikingGroup *source, NeuronGroup *destination, const char *filename, AurynFloat w_o, AurynFloat a_m, AurynFloat a_mm, AurynFloat a_p, AurynFloat a_pp, AurynFloat tau_hom, AurynFloat kappa, TransmitterType transmitter, AurynFloat kw)
        : TripletConnection(source, destination, transmitter)
{
        init(w_o, kw, tau_hom, kappa, a_m, a_mm, a_p, a_pp);
        if (! load_from_file(filename) )
                throw AurynMMFileException();
}

/*****************
 *** init crap ***
 *****************/

ZynapseConnection::~ZynapseConnection()
{
        if (dst->get_post_size() > 0)
                free();
}

void ZynapseConnection::init(AurynFloat w_o, AurynFloat k_w, AurynFloat tau_hom, AurynFloat kappa,
                             AurynFloat a_m, AurynFloat a_mm, AurynFloat a_p, AurynFloat a_pp)
{
        if (dst->get_post_size() == 0) return;

        dist = new boost::normal_distribution<> (0., 1.);
        die = new boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > (gen, *dist);
        if (!has_been_seeded)
                seed(communicator->rank());

        wo = w_o;
        kw = k_w;

        if (tau_hom)
                homeo = true;
        else
                homeo = false;

        tr_pot_pre = src->get_pre_trace(TAU_POT_PRE);
        tr_pot_post = dst->get_post_trace(TAU_POT_POST);
        if (a_mm)
                tr_dep_pre = src->get_pre_trace(TAU_DEP_PRE);
        tr_dep_post = dst->get_post_trace(TAU_DEP_POST);
        if (homeo)
                tr_hom_post = dst->get_post_trace(tau_hom);

        set_plast_constants(a_m, a_pp, a_mm, a_p);
        if (homeo)
                hom_cte = 0.5/pow(kappa*tau_hom, PZ);

        stdp_active = true;

        euler[0] = TUPD/TAUX;
        euler[1] = TUPD/TAUYZ;
        euler[2] = TUPD/TAUYZ;

        t_updates = TUPD/dt;

        eta = sqrt(ETAXYZ*TUPD);

        set_name("ZynapseConnection");
}

void ZynapseConnection::free()
{
        if (dst->get_post_size() == 0) return;

        delete dist;
        delete die;
}

/************
 *** body ***
 ************/

void ZynapseConnection::integrate()
{
        for (int layer=0; layer!=3; layer++) {
                gsl_vector_float *x = layers[layer];

                // X^3-X = X*(X*X-1)
                gsl_blas_scopy(x,bufs[0]);
                auryn_vector_float_mul(bufs[0],x);
                auryn_vector_float_add_constant(bufs[0],-1.);
                auryn_vector_float_mul(bufs[0],x);
                if (layer==0) {
                        // -meta*(1-dxy)*(y-x)
                        gtogsl(bufs[1]);
                        auryn_vector_float_add_constant(bufs[1],-1.);
                        auryn_vector_float_mul(bufs[1],diffs[0]);
                        // auryn_vector_float_saxpy(-TILT,bufs[1],bufs[0]); // for noise tuning only !!
                        auryn_vector_float_saxpy(-META_YX,bufs[1],bufs[0]);
                }
                if (layer==1) {
                        // -meta*(1-dyz)*(z-y)
                        auryn_vector_float_saxpy(META_ZY*(1.-dst->get_protein()),diffs[1],bufs[0]);
                        // -tilt*dxy*(x-y)
                        gtogsl(bufs[1]);
                        auryn_vector_float_mul(bufs[1],diffs[0]);
                        auryn_vector_float_saxpy(-TILT,bufs[1],bufs[0]);
                }
                if (layer==2)
                        // -tilt*dyz*(y-z)
                        auryn_vector_float_saxpy(-TILT*dst->get_protein(),diffs[1],bufs[0]);
                // add (t-to)/tau*(-Xdot) to X
                auryn_vector_float_saxpy(-euler[layer],bufs[0],x);
                // noise
                noise(layer);
        }
        // diffs and couplings
        compute_diffs();
        dst->update_prot();
}

inline void ZynapseConnection::noise(NeuronID layer)
{
        float *data_begin = layers[layer]->data;
        for (float *dat=data_begin;
             dat<(data_begin+get_nonzero()); dat++)
                *dat += eta*(*die)();
}

inline void ZynapseConnection::compute_diffs_loc(NeuronID i)
{
        float x = layers[0]->data[i],
                y = layers[1]->data[i],
                z = layers[2]->data[i];
        diffs[0]->data[i] = x-y;
        diffs[1]->data[i] = y-z;
}

inline void ZynapseConnection::compute_diffs(NeuronID layer)
{
        gsl_blas_scopy(layers[layer], diffs[layer]);
        auryn_vector_float_sub(diffs[layer], layers[layer+1]);
}

inline void ZynapseConnection::compute_diffs()
{
        compute_diffs(0);
        compute_diffs(1);
}

inline void ZynapseConnection::dw_dep(AurynWeight *x, NeuronID i, NeuronID pre, NeuronID post)
{
        NeuronID translated_spike = dst->global2rank(post); // only to be used for post traces
        AurynWeight deppost = tr_dep_post->get(translated_spike),
                deppre = 0, dw,
                reset = gsl_vector_float_get(layers[2],i)-*x;
        if (a3m) deppre = tr_dep_pre->get(pre);
        dw = deppost*(a2m+a3m*deppre);
        if (homeo) reset *= pow(tr_hom_post->get(translated_spike), PZ)*hom_cte;
        if (reset<0) dw *= 1-CR*reset;
        if (dw>1) dw = 1;
        if (reset>0) tr_gxy->add(i, dw*(1.-tr_gxy->get(i)));
        dw *= (1+*x);
        *x -= dw;
}

inline void ZynapseConnection::dw_pot(NeuronID i, NeuronID pre, NeuronID post)
{
        // post translation is done in loop below
        AurynWeight potpre = tr_pot_pre->get(pre),
                potpost = tr_pot_post->get(post),
                dw,
                *x = gsl_vector_float_ptr(layers[0],i),
                reset = gsl_vector_float_get(layers[2],i)-*x;
        dw = potpre*(a2p+a3p*potpost);
        if (homeo) reset *= pow(tr_hom_post->get(post), PZ)*hom_cte;
        if (reset>0) dw *= 1+CR*reset;
        if (dw>1) dw = 1;
        if (reset<0) tr_gxy->add(i,dw*(1.-tr_gxy->get(i)));
        dw *= (1-*x);
        *x += dw;
}

inline void ZynapseConnection::propagate_forward()
{
        for (int i = 0 ; i < sys->get_com()->size() ; ++i) {
                for (SpikeContainer::const_iterator spike = src->get_spikes(i)->begin() ; // spike = pre_spike
                     spike != src->get_spikes(i)->end() ; ++spike ) {
                        for (NeuronID * c = w->get_row_begin(*spike) ; c != w->get_row_end(*spike) ; ++c ) { // c = post index
                                NeuronID k = c-fwd_ind;
                                AurynWeight *x = gsl_data+k;
                                transmit( *c , wo/2.*(kw+1+(kw-1)*(*x)));
                                if ( stdp_active )
                                        dw_dep(x, k, *spike, *c);
                        }
                }
        }
}

inline void ZynapseConnection::propagate_backward()
{
        SpikeContainer::const_iterator spikes_end = dst->get_spikes_immediate()->end();
        // process spikes
        for (SpikeContainer::const_iterator spike = dst->get_spikes_immediate()->begin(); spike != spikes_end ; ++spike ) { // spike = post_spike
                NeuronID translated_spike = dst->global2rank(*spike); // only to be used for post traces
                for (NeuronID * c = bkw->get_row_begin(*spike) ; c != bkw->get_row_end(*spike) ; ++c ) {
                        NeuronID i = bkw_data[c-bkw_ind]-fwd_data;
                        dw_pot(i, *c, translated_spike);
                }
        }
}

void ZynapseConnection::propagate()
{
        propagate_forward();
        if (stdp_active) propagate_backward();
}

void ZynapseConnection::evolve()
{
        AurynTime current = sys->get_clock();
        if (current%t_updates==0)
                integrate();
}

AurynWeight ZynapseConnection::wtox(AurynWeight value)
{
        return (value*2/wo-(kw+1))/(kw-1);
}

void ZynapseConnection::set_hom_trace(AurynFloat freq)
{
        if ( dst->get_post_size() > 0 )
                tr_hom_post->setall(freq*tr_hom_post->get_tau());
}

AurynWeight ZynapseConnection::get_homeo(NeuronID i)
{
        return tr_hom_post->get(i)/tr_hom_post->get_tau();
}

void ZynapseConnection::random_data_potentiation(AurynFloat z_up, bool reset)
{
        if (reset) {
                set_x(-1); set_y(-1); set_z(-1);
        }
        if (z_up) {
                boost::exponential_distribution<> exp_dist(z_up);
                boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > exp_die(gen, exp_dist);

                AurynLong x = (AurynLong) exp_die(), y;
                while (x < get_nonzero()) {
                        potentiate(x);
                        y = (AurynLong)(exp_die()+0.5);
                        x += (y==0)?1:y;
                }
        }
}

void ZynapseConnection::count_states(AurynInt *states)
{
        float *data[3];
        for (int j=0; j<3; j++)
                data[j] = layers[j]->data;
        for (NeuronID i=0; i<get_nonzero(); i++,data[0]++,data[1]++,data[2]++) {
                int state = 0;
                for (int j=0; j<3; j++)
                        if ((*data[j])>0)
                                state += pow(2,j);
                states[state]++;
        }
}

bool ZynapseConnection::write_to_file(const char * filename)
{
        ofstream outfile;
        outfile.open(filename, ios::out);
        if (!outfile) {
                stringstream oss;
                oss << "Can't open output file " << filename;
                logger->msg(oss.str(),ERROR);
                throw AurynOpenFileException();
                return false;
        }
        outfile << "%%Zynapse matrix\n"
                << "% Auryn weight matrix. Has to be kept in row major order for load operation.\n"
                << "% data order: i j early tag z g_xy\n"
                << "% Connection name: " << get_name() << "\n"
                << "% Locked range: " << dst->get_locked_range() << "\n"
                << "% destination protein level: " << dst->get_protein() << "\n"
                << "%\n"
                << get_m_rows() << " " << get_n_cols() << " " << w->get_nonzero() << endl;

        float *x, *y, *z;
        if (get_nonzero()) {
                integrate();
                x = gsl_data;
                y = layers[1]->data;
                z = layers[2]->data;
        }
        for ( NeuronID i = 0 ; i < get_m_rows() ; ++i ) {
                for ( NeuronID * j = w->get_row_begin(i) ; j != w->get_row_end(i) ; ++j ) {
                        int n = j-fwd_ind;
                        outfile << i+1 << " " << *j+1 << " " << scientific
                                << *(x+n) << " " << *(y+n) << " " << *(z+n)
                                << " " << tr_gxy->get(n) << fixed << "\n";
                }
        }

        outfile.close();
        return true;
}

bool ZynapseConnection::load_from_file(const char * filename)
{
        char buffer[256];
        ifstream infile (filename);
        if (!infile) {
                stringstream oss;
                oss << "Can't open input file " << filename;
                logger->msg(oss.str(),ERROR);
                return false;
        }

        set_name(filename);

        AurynInt i,j,k;
        AurynInt count = 0;
        AurynFloat early, tag, zed, coupling;

        bool zynapse_type = true;
        // read connection details
        infile.getline (buffer,100); count++;
        string header("%%Zynapse matrix");
        if (header.compare(buffer)!=0) {
                header = "%%MatrixMarket matrix coordinate real general";
                if (header.compare(buffer)!=0) {
                        stringstream oss;
                        oss << "Input format not recognized.";
                        logger->msg(oss.str(),ERROR);
                        return false;
                }
                zynapse_type = false;
        }

        while ( buffer[0]=='%' ) {
                infile.getline (buffer,100);
                count++;
        }

        sscanf (buffer,"%u %u %u",&i,&j,&k);

        set_size(i,j);

        if (has_been_allocated) {
                if (w->get_datasize() > k) {
                        fwd->clear();
                        bkw->clear();
                }
                else {
                        fwd->resize_buffer_and_clear(k);
                        bkw->resize_buffer_and_clear(k);
                }
                dealloc_vectors();
        } else {
                allocate(k);
                fwd = w;
                bkw = new BackwardMatrix(j, i, k);
        }

        stringstream oss;
        oss << get_name() << ": Reading from file ("<< get_m_rows()<<"x"<<get_n_cols()<<" @ "<<1.*k/(src->get_size()*dst->get_rank_size())<<")";
        logger->msg(oss.str(),NOTIFICATION);

        while ( infile.getline (buffer,100) )
                {
                        count++;
                        if (zynapse_type)
                                sscanf (buffer,"%u %u %e %e %e %e", &i, &j, &early, &tag, &zed, &coupling);
                        else {
                                sscanf (buffer,"%u %u %e", &i, &j, &early);
                                early = wtox(early);
                        }
                        try {
                                if (push_back(i-1, j-1, early) )
                                        count++;
                        }
                        catch ( AurynMatrixPushBackException )
                                {
                                        stringstream oss;
                                        oss << "Push back failed. Error in line=" << count << ", "
                                            << " i=" << i
                                            << " j=" << j
                                            << " v=" << early << ". "
                                            << "Bad row major order?";
                                        logger->msg(oss.str(),ERROR);
                                        throw AurynMMFileException();
                                        return false;
                                }
                        catch ( AurynMatrixBufferException )
                                {
                                        stringstream oss;
                                        oss << get_name() <<": Buffer full after pushing " << count << " elements."
                                            << " There are pruned connections!";
                                        logger->msg(oss.str(),ERROR);
                                        return false;
                                }
                }

        finalize();

        if (zynapse_type && dst->evolve_locally()) {
                infile.clear();
                infile.seekg(0,ios::beg);
                infile.getline(buffer,100);
                while ( buffer[0]=='%' )
                        infile.getline(buffer,100);

                float *x = gsl_data,
                        *y = layers[1]->data,
                        *z = layers[2]->data,
                        *c = tr_gxy->get_state_ptr();
                while ( infile.getline (buffer,100) )
                        {
                                sscanf (buffer,"%u %u %e %e %e %e",
                                        &i, &j, &early, &tag, &zed, &coupling);
                                NeuronID n = w->get_ptr(i-1,j-1)-fwd_data;
                                *(x+n) = early;
                                *(y+n) = tag;
                                *(z+n) = zed;
                                *(c+n) = coupling;
                        }
        }

        infile.close();

        return true;
}

void ZynapseConnection::seed(int s)
{
#ifdef RANDOM_SEED
        timeval ss;
        gettimeofday(&ss,NULL);
        s += ss.tv_usec-100*(ss.tv_usec/100);
#endif
        gen.seed(s);
        has_been_seeded = true;
}

// NeuronID ZynapseConnection::get_size() TODO: who uses that?
// {
//   return size;
// }

void ZynapseConnection::stats(AurynFloat &mean, AurynFloat &std)
{
        NeuronID count = get_nonzero();
        AurynFloat sum = 0;
        AurynFloat sum2 = 0;
        float *x = gsl_data;
        for ( NeuronID i = 0 ; i < count ; ++i,++x ) {
                sum += *x;
                sum2 += pow(*x,2);
        }
        if ( count <= 1 ) {
                mean = sum;
                std = 0;
                return;
        }
        mean = sum/count;
        std = sqrt(sum2/count-pow(mean,2));
}

void ZynapseConnection::stats(AurynFloat &mean, AurynFloat &std, vector<NeuronID> * presynaptic_list)
{
        NeuronID count = 0;
        AurynFloat sum = 0;
        AurynFloat sum2 = 0;
        vector<NeuronID>::const_iterator i_end = presynaptic_list->end();
        NeuronID * ind = w->get_row_begin(0);
        for ( vector<NeuronID>::const_iterator i = presynaptic_list->begin(); i != i_end; ++i)
                for ( NeuronID * c = w->get_row_begin(*i); c != w->get_row_end(*i); ++c) {
                        AurynWeight value = gsl_data[c-ind];
                        sum += value;
                        sum2 += pow(value,2);
                        ++count;
                }
        if ( count <= 1 ) {
                mean = sum;
                std = 0;
                return;
        }
        mean = sum/count;
        std = sqrt(sum2/count-pow(mean,2));
}

// float * ZynapseConnection::get_layer_ptr(int layer) // TODO: who?
// {
//   return layers[layer]->data;
// }

void ZynapseConnection::set_noise(AurynFloat level)
{
        eta = sqrt(level*t_updates*dt);
}

void ZynapseConnection::set_tau(AurynInt layer, AurynFloat level)
{
        euler[layer] = TUPD/level;
}

AurynWeight ZynapseConnection::get(NeuronID i, NeuronID j)
{
        return *get_ptr(i,j);
}

AurynWeight ZynapseConnection::get_data(NeuronID i)
{
        return gsl_vector_float_get(layers[0],i);
}

AurynWeight * ZynapseConnection::get_ptr(NeuronID i, NeuronID j)
{
        if (w->get_ptr(i,j) == NULL) return NULL;
        NeuronID ind = w->get_ptr(i,j)-fwd_data;
        return gsl_data+ind;
}

void ZynapseConnection::set_data(NeuronID i, AurynWeight value) // TODO: someone using set_x?
{
        gsl_vector_float_set(layers[0],i,value);
        compute_diffs_loc(i);
}

void ZynapseConnection::set(vector<neuron_pair> element_list, AurynWeight value)
{
        for (vector<neuron_pair>::iterator iter = element_list.begin() ; iter != element_list.end() ; ++iter)
                set((*iter).i,(*iter).j,value);
}

void ZynapseConnection::set(NeuronID i, NeuronID j, AurynWeight value)
{
        value = max(value,get_wmin());
        AurynWeight *ptr = get_ptr(i,j);
        *ptr = value;
}

void ZynapseConnection::set_x(AurynWeight value)
{
        if (!dst->evolve_locally() ) return;
        gsl_vector_float_set_all(layers[0],value);
        compute_diffs(0);
}

void ZynapseConnection::set_y(AurynFloat level)
{
        if (!dst->evolve_locally() ) return;
        gsl_vector_float_set_all(layers[1],level);
        compute_diffs();
}

void ZynapseConnection::set_z(AurynFloat level)
{
        if (!dst->evolve_locally() ) return;
        gsl_vector_float_set_all(layers[2],level);
        compute_diffs(1);
}

void ZynapseConnection::potentiate_bkw(NeuronID j)
{
        for (NeuronID * k = bkw->get_row_begin(j); k != bkw->get_row_end(j); ++k) {
                int i = bkw_data[k-bkw_ind]-fwd_data;
                potentiate(i);
        }
}

void ZynapseConnection::potentiate_bkw_range(NeuronID n)
{
        for (NeuronID k = 0; k != n; ++k)
                potentiate_bkw(k);
}

void ZynapseConnection::potentiate(NeuronID i)
{
        for (int layer=0; layer<3; layer++)
                gsl_vector_float_set(layers[layer],i,1.);
        compute_diffs_loc(i);
}

void ZynapseConnection::potentiate()
{
        for (int layer=0; layer<3; layer++)
                gsl_vector_float_set_all(layers[layer],1.);
        compute_diffs();
}

void ZynapseConnection::decay(NeuronID i)
{
        float z = gsl_vector_float_get(layers[2], i);
        for (int layer=0; layer<2; layer++)
                gsl_vector_float_set(layers[layer],i,z);
        compute_diffs_loc(i);
}

void ZynapseConnection::decay()
{
        gsl_blas_scopy(layers[2],layers[1]);
        gsl_blas_scopy(layers[2],layers[0]);
        compute_diffs();
}

void ZynapseConnection::rebounce(NeuronID i)
{
        float y = gsl_vector_float_get(layers[1], i);
        gsl_vector_float_set(layers[0],i,y);
        compute_diffs_loc(i);
}

void ZynapseConnection::rebounce()
{
        gsl_blas_scopy(layers[1],layers[0]);
        compute_diffs();
}

void ZynapseConnection::stabilize(NeuronID i)
{
        float x = gsl_vector_float_get(layers[0], i);
        for (int layer=1; layer<3; layer++)
                gsl_vector_float_set(layers[layer],i,x);
        compute_diffs_loc(i);
}

void ZynapseConnection::stabilize()
{
        gsl_blas_scopy(layers[0],layers[1]);
        gsl_blas_scopy(layers[0],layers[2]);
        compute_diffs();
}

void ZynapseConnection::fall(NeuronID i)
{
        for (int n=0; n<3; n++) {
                float x = gsl_vector_float_get(layers[n], i), xx;
                if (x>0) xx = 1;
                else xx = -1;
                gsl_vector_float_set(layers[n], i, xx);
        }
        compute_diffs_loc(i);
}

void ZynapseConnection::fall()
{
        float *x = layers[0]->data,
                *y = layers[1]->data,
                *z = layers[2]->data;
        for (NeuronID i=0; i<get_nonzero(); i++,x++,y++,z++) {
                if (*x>0) *x = 1;
                else *x = -1;
                if (*y>0) *y = 1;
                else *y = -1;
                if (*z>0) *z = 1;
                else *z = -1;
        }
        compute_diffs();
}

void ZynapseConnection::tag(NeuronID i)
{
        if (abs(gsl_vector_float_get(diffs[1], i))<1) {
                float x = gsl_vector_float_get(layers[0], i);
                gsl_vector_float_set(layers[1],i,x);
                tr_gxy->set(i, 0);
        }
        compute_diffs_loc(i);
}

void ZynapseConnection::tag()
{
        float *x = layers[0]->data,
                *y = layers[1]->data,
                *d = diffs[1]->data;
        for (NeuronID i=0; i<get_nonzero(); i++,x++,y++,d++)
                if (abs(*d)<1) {
                        *y = *x;
                        tr_gxy->set(i, 0);
                }
        compute_diffs();
}

void ZynapseConnection::depress()
{
        for (int layer=0; layer<3; layer++)
                gsl_vector_float_set_all(layers[layer],-1.);
        compute_diffs();
}

AurynFloat ZynapseConnection::get_g(NeuronID i)
{
        return tr_gxy->get(i);
}

void ZynapseConnection::set_g(AurynFloat value)
{
        if (!dst->evolve_locally() ) return;
        tr_gxy->setall(value);
}

inline void ZynapseConnection::gtogsl(gsl_vector_float *vector)
{
        tr_gxy->update();
        float *gxy = tr_gxy->get_state_ptr(),
                *vect = vector->data;
        for (NeuronID i=0; i!=get_nonzero(); i++,gxy++,vect++)
                if (*gxy>=THETAG) *vect = 1.;
                else *vect = 0;
        // *vect = *gxy;
}

void ZynapseConnection::set_plast_constants(AurynFloat a_m, AurynFloat a_pp, AurynFloat a_mm, AurynFloat a_p)
{
        a2m = a_m/TAU_DEP_POST;
        a3m = a_mm/TAU_DEP_POST/TAU_DEP_PRE;
        a2p = a_p/TAU_POT_PRE;
        a3p = a_pp/TAU_POT_POST/TAU_POT_PRE;
}
