/**
 * ml-dataset-metadata - a class to manipulate metadata for dataset
 * @version v0.0.0
 * @link https://github.com/mljs/dataset-metadata#readme
 * @license MIT
 */
(function (global, factory) {
	typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
	typeof define === 'function' && define.amd ? define(['exports'], factory) :
	(global = global || self, factory(global.mlDatasetMetadata = {}));
}(this, (function (exports) { 'use strict';

	function unwrapExports (x) {
		return x && x.__esModule && Object.prototype.hasOwnProperty.call(x, 'default') ? x['default'] : x;
	}

	function createCommonjsModule(fn, module) {
		return module = { exports: {} }, fn(module, module.exports), module.exports;
	}

	var lib = createCommonjsModule(function (module, exports) {

	  Object.defineProperty(exports, '__esModule', {
	    value: true
	  }); // import Matrix from 'ml-matrix';

	  /**
	   * create new metadata object from 2D array
	   * @param {Array} values - a 2d array with metadata values
	   * @param {Object} [options]
	   * @param {Array} [options.headers=[...Array(columns).keys()]] - an array with metadata headers
	   * @param {Array} [options.IDs=[...Array(rows).keys()]] - an array with IDs
	   */

	  class METADATA {
	    constructor(values) {
	      let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
	      let {
	        columns,
	        rows
	      } = [];

	      if (values === true) {
	        const metadata = options;
	        this.headers = metadata.headers;
	        this.IDs = metadata.IDs;
	        this.values = options.values;
	      } else {
	        columns = values.length;
	        rows = values[0].length;
	        this.values = values;
	        let {
	          headers = [...Array(columns).keys()].map(x => (x + 1).toString()),
	          IDs = [...Array(rows).keys()].map(x => (x + 1).toString())
	        } = options;
	        this.headers = headers;
	        this.IDs = IDs;
	      }
	    }
	    /**
	     * load metadata
	     * @param {Boolean} [metadata=true] - a boolean
	     * @param {Object} [options]
	     * @param {JSON} [options.metadata] - a metadata object
	     * @return {METADATA} - a metdata object
	     */


	    static load(metadata) {
	      if (typeof metadata.name !== 'string') {
	        throw new TypeError('metadata must have a name property');
	      }

	      if (metadata.name !== 'metadata') {
	        throw new RangeError("invalid model: ".concat(metadata.name));
	      }

	      return new METADATA(true, metadata);
	    }
	    /**
	     * save metadata to JSON
	     * @return {JSON} - a JSON with metadata
	     */


	    toJSON() {
	      return {
	        name: 'metadata',
	        headers: this.headers,
	        IDs: this.IDs,
	        values: this.values
	      };
	    }
	    /**
	       * listMetadata
	       * @return {Array} - an array with headers
	       */


	    list() {
	      return this.headers;
	    }
	    /**
	       * add metadata
	       * @param {Array} value - an array with metadata
	       * @param {String} [by = 'column'] - select by row or by column
	       * @param {Object} [options]
	       * @param {String} [options.header] - a header for new metadata
	       */


	    append(values) {
	      let by = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 'column';
	      let options = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};

	      if (by === 'column') {
	        let {
	          header = (this.headers.length + 1).toString()
	        } = options;

	        if (typeof header !== 'string') {
	          console.warn('header was coerced to string');
	          header = header.toString();
	        }

	        if (this.headers.includes(header)) {
	          throw new Error('this header already exist');
	        }

	        if (values.length === this.values[0].length) {
	          this.values.push(values);
	          this.headers.push(header);
	        } else {
	          throw new Error('dimension doesn\'t match');
	        }
	      } else if (by === 'row') {
	        let {
	          ID = (this.IDs.length + 1).toString()
	        } = options;

	        if (typeof ID !== 'string') {
	          console.warn('ID was coerced to string');
	          ID = ID.toString();
	        }

	        if (this.IDs.includes(ID)) {
	          throw new Error('this ID already exist');
	        }

	        if (values.length === this.values.length) {
	          this.values.map((x, idx) => x.push(values[idx]));
	          this.IDs.push(ID);
	        } else {
	          throw new Error('dimension doesn\'t match');
	        }
	      }

	      return this;
	    }
	    /**
	     * remove row or column by index or name
	     * @param {any} index - an index or a column/row name
	     * @param {String} [by = 'row'] - select by row or by column
	     */


	    remove(index) {
	      let by = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 'row';

	      if (typeof index !== 'object') {
	        index = [index];
	      }

	      if (by === 'column') {
	        index.forEach((el, idx) => {
	          if (typeof el === 'number') {
	            index[idx] = this.headers[el];
	          }
	        });
	        index.forEach((el, idx) => {
	          let id = this.headers.indexOf(index[idx]);

	          if (id > -1) {
	            this.headers.splice(id, 1);
	            this.values.splice(id, 1);
	          }
	        });
	      } else if (by === 'row') {
	        index.forEach((el, idx) => {
	          if (typeof el === 'number') {
	            index[idx] = this.IDs[el];
	          }
	        });
	        index.forEach((el, idx) => {
	          let id = this.IDs.indexOf(index[idx]);

	          if (id > -1) {
	            this.IDs.splice(id, 1);
	            this.values.map(x => x.splice(id, 1));
	          }
	        });
	      }

	      return this;
	    }
	    /**
	       *
	       * @param {String} title - a title
	       * @return {Object} return { title, groupIDs, nClass, classVector, classFactor, classMatrix }
	       */


	    get(header) {
	      let index = this.headers.indexOf(header);
	      let classVector = this.values[index];
	      return classVector;
	    }

	    summary(header) {
	      let index = this.headers.indexOf(header);
	      let classVector = this.values[index];
	      let nObs = classVector.length;
	      let type = typeof classVector[0];
	      let counts = {};

	      switch (type) {
	        case 'string':
	          counts = summaryAClass(classVector);
	          break;

	        case 'number':
	          classVector = classVector.map(x => x.toString());
	          counts = summaryAClass(classVector);
	          break;
	      }

	      let groupIDs = Object.keys(counts);
	      let nClass = groupIDs.length; // let classFactor = classVector.map((x) => groupIDs.indexOf(x));

	      return {
	        class: header,
	        groups: counts,
	        nObs,
	        nClass
	      };
	    }

	    sample(header) {
	      let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
	      const {
	        fraction = 0.8
	      } = options;
	      let classVector = this.get(header, 'string');
	      let {
	        trainIndex,
	        testIndex,
	        mask
	      } = sampleAClass(classVector, fraction);
	      return {
	        trainIndex,
	        testIndex,
	        mask,
	        classVector
	      };
	    }

	  }

	  function summaryAClass(classVector) {
	    let counts = {};
	    classVector.forEach(x => {
	      counts[x] = (counts[x] || 0) + 1;
	    });
	    return counts;
	  }

	  function sampleAClass(classVector, fraction) {
	    // sort the vector
	    let classVectorSorted = JSON.parse(JSON.stringify(classVector));
	    let result = Array.from(Array(classVectorSorted.length).keys()).sort((a, b) => classVectorSorted[a] < classVectorSorted[b] ? -1 : classVectorSorted[b] < classVectorSorted[a] | 0);
	    classVectorSorted.sort((a, b) => a < b ? -1 : b < a | 0); // counts the class elements

	    let counts = summaryAClass(classVectorSorted);
	    console.log('counts', counts); // pick a few per class

	    let indexOfSelected = [];
	    Object.keys(counts).forEach((e, i) => {
	      let shift = [];
	      Object.values(counts).reduce((a, c, i) => shift[i] = a + c, 0);
	      console.log(shift);
	      let arr = [...Array(counts[e]).keys()];
	      let r = [];

	      for (let i = 0; i < Math.floor(counts[e] * fraction); i++) {
	        let n = arr[Math.floor(Math.random() * arr.length)];
	        r.push(n);
	        let ind = arr.indexOf(n);
	        arr.splice(ind, 1);
	      }

	      if (i === 0) {
	        indexOfSelected = indexOfSelected.concat(r);
	      } else {
	        indexOfSelected = indexOfSelected.concat(r.map(x => x + shift[i - 1]));
	      }
	    }); // sort back the index

	    let trainIndex = [];
	    indexOfSelected.forEach(e => trainIndex.push(result[e]));
	    let testIndex = [];
	    let mask = [];
	    classVector.forEach((el, idx) => {
	      if (trainIndex.includes(idx)) {
	        mask.push(true);
	      } else {
	        mask.push(false);
	        testIndex.push(idx);
	      }
	    });
	    return {
	      trainIndex,
	      testIndex,
	      mask
	    };
	  }

	  exports.METADATA = METADATA;
	});
	var index = unwrapExports(lib);
	var lib_1 = lib.METADATA;

	exports.METADATA = lib_1;
	exports.default = index;

	Object.defineProperty(exports, '__esModule', { value: true });

})));
//# sourceMappingURL=ml-dataset-metadata.js.map
