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

#include "PRPGroup.h"

using namespace auryn;


PRPGroup::PRPGroup( NeuronID size, AurynFloat load, NeuronID total )
        : AIFGroup(size,load,total)
{
        if ( evolve_locally() ) init();
}

void PRPGroup::init()
{
        tau_thr = 5e-3;

        /* for frey's experiment:
         *  tau = 250 ms
         *  dg = 10
         * else:
         *  dg = 1 */
        tau_adapt1 = 250e-3;
        dg_adapt1 = 10.;

        calculate_scale_constants();

        prp = get_state_variable("prp");
        *prp = 0;

        tau_prp_up = 1.;
        tau_prp_down = 7200.;

        prp_timestamp = 0;
        dopamine = false;
}

void PRPGroup::free()
{
}

PRPGroup::~PRPGroup()
{
        if ( evolve_locally() ) free();
}

void PRPGroup::update_prp() {
        if (!evolve_locally()) return;
	
        AurynTime timediff = *clock_ptr-prp_timestamp;
        if (timediff<=0) return;
        if (dopamine)
                *prp = (*prp-1.)*exp(-dt*timediff/tau_prp_up)+1.;
        else
                *prp *= exp(-dt*timediff/tau_prp_down);
        prp_timestamp = *clock_ptr;

}

void PRPGroup::dopamine_on()
{
        update_prp();
        dopamine = true;
}

void PRPGroup::dopamine_off()
{
        update_prp();
        dopamine = false;
}

void PRPGroup::set_prp(AurynState value)
{
        *prp = value;
}

AurynState PRPGroup::get_prp()
{
        update_prp();
	return *prp;
}

AurynState * PRPGroup::get_prp_ptr()
{
        update_prp();
        return prp;
}
