/*
 * Copyright 2014-2016 Friedemann Zenke, Lorric Ziegler
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

#ifndef PRPGROUP_H_
#define PRPGROUP_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "NeuronGroup.h"
#include "System.h"

namespace auryn {

        /*! \brief An extension of AIFGroup with Plasticity Related Proteins
         * whose production are triggered via neuromodulation (dopamine) */
        class PRPGroup : public AIFGroup
        {
        private:

        protected:

                AurynState *prp;
                AurynFloat tau_prp_up, tau_prp_down;
                bool dopamine;
                AurynTime prp_timestamp;

                void init();

                void update_pr();

        public:

                PRPGroup( NeuronID size, AurynFloat load = 1.0, NeuronID total = 0 );
                virtual ~PRPGroup();
                void calculate_scale_constants();


                void clear();

                void dopamine_on();
                void dopamine_off();
                void set_prp(AurynState value);
                AurynState get_prp();
                AurynState *get_prp_ptr();

        };

}

#endif /*PRPGROUP_H_*/
