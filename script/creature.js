/* 
 * creature.js
 * Created by Jorge Gonzalez, jorgonlor@gmail.com, January, 10, 2019.
 * Released under MIT License - see LICENSE file for details.
 */

"use strict";

var MAX_TIME_WITHOUT_IMPROVING = 12;

var BIAS_MUTATION_PROBABILITY = 0.05;
var BIAS_MUTATION_CHANGE = 0.08;

var DEBUG_MODE = false;
var DEBUG_TIME = 0.5;
var MANUAL_CONTROL = false;

var CreatureType = {"first":1, "clone":2, "cross":3, "mutation":4, "new":5};

class Food
{
    constructor(space, position, category)
    {
		let mass = 1;
		this.radius = 9;
		this.body = space.addBody(new cp.Body(mass, cp.momentForCircle(mass, 0, this.radius, cp.v(0, 0))));        
		this.shape = space.addShape(new cp.CircleShape(this.body, this.radius, cp.v(0, 0)));
		this.category = category;
		this.body.p = new cp.v(position.x, position.y);
		this.shape.setCollisionType(category);
		this.shape.group = category;
		this.shape.food = this;
		this.alive = true;
	}
}

class Creature
{
    constructor(space, position, category, creatureType)
    {
		let initializer = 'glorotNormal';

		this.brain = tf.tidy(() => {
			return tf.sequential({
				layers: [
				tf.layers.dense({units: 14, activation: 'linear', inputShape: [14], kernelInitializer: initializer, biasInitializer: initializer}),
				tf.layers.leakyReLU({units: 9, kernelInitializer: initializer, biasInitializer: initializer}),	
				tf.layers.leakyReLU({units: 4, kernelInitializer: initializer, biasInitializer: initializer}),		
				tf.layers.dense({units: 2, /*activation: 'linear',*/ kernelInitializer: initializer, biasInitializer: initializer})]
			});
		});

        this.radius = 5;
		this.sight_distance = 250;
		this.initPosition = new cp.v(position.x,position.y);
		var mass = 1.5;
		this.category = category;
        this.body = space.addBody(new cp.Body(mass, cp.momentForCircle(mass, 0, this.radius, cp.v(0, 0))));        
        this.shape = space.addShape(new cp.CircleShape(this.body, this.radius, cp.v(0, 0)));
        this.shape.setElasticity(0.8);
        this.shape.setFriction(0.1);
        this.shape.setCollisionType(category);
        //this.shape.setLayers(CREATURE_CATEGORY);
        this.shape.group = category;
        this.space = space;
		this.body.p = new cp.v(position.x, position.y);
		this.body.setAngle(0);
        this.shape.body.rot = cp.v(1,0);
		this.shape.creature = this;
		this.creatureType = creatureType;

		this.hitPoints = [];
		this.eyePoints = [];
        this.alive = true;

		this.lapInitTime = self.now;
		this.debugTime = self.now;
        this.timeWhenMax = self.now;
		this.maxFitness = 0.0;	

		this.angle = cp.v.toangle(this.body.rot);
		this.initAngle = this.angle;
		this.spinCount = 0;
		this.negativePropulsionCount = 0;
		
		this.showEyeTracing = true;
		this.updateCycle = 0;

		this.maxEnergy = 1700;
		this.energy = this.maxEnergy;
		this.foodCollectedCount = 0;
		this.age = 0;
    }

    calculateEyesRotations()
    {
        var rot = this.shape.body.rot;

        var rot_left_60 = cp.v(rot.x * 0.5 - rot.y * 0.8666, rot.x * 0.8666 + rot.y * 0.5);
		var rot_left_30 = cp.v(rot.x * 0.8666 - rot.y * 0.5, rot.x * 0.5 + rot.y * 0.8666);
		var rot_left_15 = cp.v(rot.x * 0.9659 - rot.y * 0.2588, rot.x * 0.2588 + rot.y * 0.9659);

		var rot_right_15 = cp.v(rot.x * 0.9659 + rot.y * 0.2588, -rot.x * 0.2588 + rot.y * 0.9659);
        var rot_right_30 = cp.v(rot.x * 0.8666 + rot.y * 0.5, -rot.x * 0.5 + rot.y * 0.8666);
        var rot_right_60 = cp.v(rot.x * 0.5 + rot.y * 0.8666, -rot.x * 0.8666 + rot.y * 0.5);

        return [rot_left_60, rot_left_30, rot_left_15, rot, rot_right_15, rot_right_30, rot_right_60];
    }

    update()
    {
		if(!this.alive) 
			return;

        var p = this.shape.body.p;
		var now = self.now;
		
		if(this.alive)
			this.age += 1;

		this.energy -= 1;
		if(this.energy <= 0) {
			this.alive = false;
			console.log("Death by starvation");
		}
		// if(this.updateCycle % 20 == 0) {
		// 	let f = this.fitness();
		// 	if(f > this.maxFitness * 1.02) {
		// 		this.maxFitness = f;
		// 		this.timeWhenMax = now;
		// 	}
		// }

        // if(now - this.timeWhenMax > MAX_TIME_WITHOUT_IMPROVING * 1000) {
        //     this.alive = false;
        //     console.log("Death by not improving");
		// }

		var new_angle = cp.v.toangle(this.body.rot);
		if(new_angle > this.initAngle && new_angle < this.initAngle + 0.4 && this.angle < this.initAngle && this.angle > this.initAngle - 0.4)
			this.spinCount += 1;
		if(new_angle < this.initAngle && new_angle > this.initAngle - 0.4 && this.angle > this.initAngle && this.angle < this.initAngle + 0.4)
			this.spinCount -= 1;

		if(this.spinCount > 2 || this.spinCount < -2) {
			this.alive = false;
			console.log("Death by spinning too much");
		}
		this.angle = new_angle;

		if(this.negativePropulsionCount > 180) {
			this.alive = false;
			console.log("Death by too negative propulsion");
		}
        
        var rots = this.calculateEyesRotations();
        
		this.hitPoints = [];
		this.eyePoints = [];
		var wall_eye_signals = [];
		var food_eye_signals = [];
        for(var i = 0; i < rots.length; ++i) {            
            var p_near = vadd(p, vmult(rots[i], this.radius));
			var p_far = vadd(p, vmult(rots[i], this.sight_distance));
			
			this.eyePoints.push([p_near, p_far]);

			var wall_eye_signal = 0.0;
			var food_eye_signal = 0.0;
            var query = this.space.segmentQueryFirst(p_near, p_far, this.shape.group, this.shape.group);
            if(query != null) {
				var hitPoint = query.hitPoint(p_near, p_far);
				this.hitPoints.push(hitPoint);
				var signal = signal = (this.sight_distance - vdist(p, hitPoint)) / this.sight_distance;
				if(query.shape.group == window.WALL_CATEGORY) {					
					wall_eye_signal = signal;
				}
				else if(query.shape.group == window.FOOD_CATEGORY) {
					food_eye_signal = signal;
				}
            }
			wall_eye_signals.push(wall_eye_signal);
			food_eye_signals.push(food_eye_signal);
		}

		if(MANUAL_CONTROL == true) return;
		
		//let vel = vlength(this.body.getVel()) / 100;
		var eye_signals = wall_eye_signals.concat(food_eye_signals);
		
        var model_output = tf.tidy(()=> {
			var p = this.brain.predict(tf.tensor2d([eye_signals]));
			if(self.sim.useTanhEdit.checked)
				return tf.tanh(p);
			else
				return p;			
		});
		
        var thrust_control = model_output.get(0,0);
        var turn_control = model_output.get(0,1);

        this.body.applyImpulse( vmult(this.shape.body.rot, 10 * thrust_control), cp.v(0, 0));

        this.body.applyImpulse( vmult(cp.v(-3, 0), turn_control), cp.v(0, 1));
		this.body.applyImpulse( vmult(cp.v(3, 0), turn_control), cp.v(0, -1));

		if(thrust_control < 0) 
			++this.negativePropulsionCount;
		else
			this.negativePropulsionCount = 0;

		
		
		if(DEBUG_MODE && now - this.debugTime > DEBUG_TIME * 1000) {
			this.debugTime = now;
			console.log("Thrust: " + thrust_control + ", Turn: " + turn_control);			
		}

		++this.updateCycle;
		if(this.updateCycle >= 10000000)
			this.updateCycle = 0;
    }
    
    clone() {
		var new_creature = new Creature(this.space, this.initPosition, this.category);
		new_creature.creatureType = CreatureType.clone;

		for(var i = 0; i < this.brain.layers.length; ++i) {

			if(this.brain.layers[i].getWeights().length == 0) continue;

			var weights = this.brain.layers[i].getWeights()[0].dataSync();
			var new_weights = new_creature.brain.layers[i].getWeights()[0].dataSync();
			for(var j = 0; j < weights.length; ++j) {
				new_weights[j] = weights[j];
			}

			var bias = this.brain.layers[i].getWeights()[1].dataSync();
			var new_bias = new_creature.brain.layers[i].getWeights()[1].dataSync();
			for(var j = 0; j < bias.length; ++j) {
				new_bias[j] = bias[j];
			}
		}

		return new_creature;
    }
	
	crossover(other) {
		for(var i = 0; i < this.brain.layers.length; ++i) {

			if(this.brain.layers[i].getWeights().length == 0) continue;

			// Weights
			var weights = this.brain.layers[i].getWeights()[0].dataSync();
			var other_weights = other.brain.layers[i].getWeights()[0].dataSync();
			var mid = weights.length / 2;
			for(var j = mid; j < weights.length; ++j) {
				weights[j] = other_weights[j];
			}

			// Bias
			var bias = this.brain.layers[i].getWeights()[1].dataSync();
			var other_bias = other.brain.layers[i].getWeights()[1].dataSync();
			mid = bias.length / 2;
			for(var j = mid; j < bias.length; ++j) {
				bias[j] = other_bias[j];
			}
		}
    }

    mutate(mutationProbability, mutationChange) {
		for(let i = 0; i < this.brain.layers.length; ++i) {

			if(this.brain.layers[i].getWeights().length == 0) continue;

			var weights = this.brain.layers[i].getWeights()[0].dataSync();
			for(let j = 0; j < weights.length; ++j) {
				if(Math.random() < mutationProbability) {
					weights[j] += Math.random() * mutationChange - (mutationChange / 2.0);
					// if(weights[j] > 1.3) weights[j] = 1.3;
					// if(weights[j] < -1.3) weights[j] = -1.3;
				}
			}

			var bias = this.brain.layers[i].getWeights()[1].dataSync();
			for(let j = 0; j < bias.length; ++j) {
				if(Math.random() < BIAS_MUTATION_PROBABILITY) {
					bias[j] += Math.random() * BIAS_MUTATION_CHANGE - (BIAS_MUTATION_CHANGE / 2.0);
					// if(bias[j] > 1.3) bias[j] = 1.3;
					// if(bias[j] < -1.3) bias[j] = -1.3;
				}
			}
		}
		return this;
    }

    fitness() {
		return 10 * this.age + this.foodCollectedCount * this.maxEnergy + this.energy;
	}
	
	testFastPointInShape(p) {
		if(Math.abs(p.x - this.body.p.x) < this.radius && Math.abs(p.y - this.body.p.y) < this.radius)
			return true;
		return false;
	}
}