const assert = require('node:assert/strict');
const {spawn} = require('node:child_process');
const {chromium} = require(process.cwd() + '/.browser-test/node_modules/playwright');

const port = 8765;
const server = spawn('python', ['-m','http.server',String(port),'--bind','127.0.0.1','--directory','dist'], {stdio:'ignore'});

(async () => {
  let browser;
  try {
    browser = await chromium.launch({headless:true, args:['--no-sandbox']});
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${port}/lab/`,{waitUntil:'domcontentloaded',timeout:45000});
    console.log('JupyterLite page loaded:', await page.title());
    const outcome = await page.evaluate(async (port) => {
      const src=`http://127.0.0.1:${port}/vendor/tf.min.js`;
      const code=`
        self.onmessage = async ({data}) => {
          try {
            const {loadPyodide} = await import(data.pyodide);
            const py = await loadPyodide({indexURL:data.pyodideBase});
            // Isolated backend probe: do not install Jupyter/ipywidgets here.
            // An unconstrained browser-side PyPI resolver can pull new
            // ipykernel/tornado wheels unsupported by old Pyodide. Real
            // widget + pinned IPython are tested in JupyterLite REPL instead.
            await py.loadPackage(['numpy','pandas','scipy','matplotlib','scikit-learn']);
            await py.loadPackage(['h5py', 'Pillow']);
            await py.runPythonAsync(
                "import h5py\\n"
              + "from PIL import Image\\n"
              + "import numpy as np\\n"
              + "with h5py.File('/tmp/ml-wheel-probe.h5', 'w') as f:\\n"
              + "    f['values'] = np.array([2, 3, 5])\\n"
              + "with h5py.File('/tmp/ml-wheel-probe.h5', 'r') as f:\\n"
              + "    assert f['values'][:].tolist() == [2, 3, 5]\\n"
              + "assert Image.new('RGB', (3, 2)).size == (3, 2)\\n"
            );
            const script = await (await fetch(data.bridge)).text();
            py.FS.writeFile('/home/pyodide/tfjs_bridge.py', script);
            py.FS.mkdirTree('/home/pyodide/browser_ml');
            py.FS.writeFile('/home/pyodide/browser_ml/__init__.py', '');
            py.FS.writeFile('/home/pyodide/browser_ml/tfjs_bridge.py', script);
            const compatText = await (await fetch(data.compat)).text();
            py.FS.writeFile('/home/pyodide/browser_ml/tfjs_compat.py', compatText);
            const tensorText = await (await fetch(data.tensor)).text();
            py.FS.writeFile('/home/pyodide/browser_ml/tfjs_tensor.py', tensorText);
            const layersText = await (await fetch(data.layers)).text();
            py.FS.writeFile('/home/pyodide/browser_ml/tfjs_layers.py', layersText);
            const output = await py.runPythonAsync(
                "import numpy as np, scipy, pandas, matplotlib, sklearn\\n"
              + "from sklearn.linear_model import LinearRegression\\n"
              + "reg=LinearRegression().fit([[1],[2],[3]],[3,5,7])\\n"
              + "assert abs(reg.predict([[4]])[0]-9)<1e-3\\n"
              + "import tfjs_bridge\\n"
              + "await tfjs_bridge.load_tfjs('"+data.tfjs+"', backend='cpu')\\n"
              + "assert tfjs_bridge.backend() == 'cpu'\\n"
              + "await tfjs_bridge.load_tfjs('"+data.tfjs+"', backend='auto')\\n"
              + "assert tfjs_bridge.backend() in ('webgpu', 'webgl', 'cpu')\\n"
              + "model=tfjs_bridge.Sequential([tfjs_bridge.Dense(4,'relu',input_shape=(2,)),tfjs_bridge.Dense(1,'sigmoid')])\\n"
              + "model.compile(loss='binaryCrossentropy',learning_rate=0.05)\\n"
              + "X=np.asarray([[0,0],[0,1],[1,0],[1,1]]*5,dtype='float32')\\n"
              + "y=np.asarray([[0],[1],[1],[1]]*5,dtype='float32')\\n"
              + "h=await model.fit(X,y,epochs=12,batch_size=8)\\n"
              + "pred=model.predict(X)\\n"
              + "assert pred.shape == y.shape\\n"
              + "print('Pyodide WASM scientific imports PASS; TensorFlow.js worker training PASS')\\n"
              + "str(tfjs_bridge.backend_info())"
            );
            globalThis.eval(await (await fetch(data.losses)).text());
            await py.runPythonAsync(
                "from browser_ml.tfjs_compat import install\\n"
              + "tf = await install(backend='cpu')\\n"
              + "from tensorflow.keras.models import Sequential\\n"
              + "from tensorflow.keras.layers import Dense\\n"
              + "from tensorflow.keras.activations import sigmoid\\n"
              + "import numpy as np\\n"
              + "norm = tf.keras.layers.Normalization(axis=-1)\\n"
              + "norm.adapt(np.asarray([[1,2],[3,4]], dtype='float32'))\\n"
              + "assert np.isfinite(norm(np.asarray([[2,3]]))).all()\\n"
              + "assert abs(float(sigmoid(0).numpy())-0.5)<1e-5\\n"
              + "model2 = Sequential([tf.keras.Input(shape=(2,)), Dense(3,'relu',name='layer1'), Dense(1,'sigmoid',name='layer2')])\\n"
              + "weights = model2.get_layer('layer1').get_weights()\\n"
              + "assert len(weights)==2 and weights[0].shape==(2,3), f'Dense weights unexpected: {len(weights)}, {[w.shape for w in weights]}'\\n"
              + "model2.get_layer('layer1').set_weights(weights)\\n"
              + "model2.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))\\n"
              + "hist2 = await model2.fit([[0,0],[0,1],[1,0],[1,1]],[[0],[1],[1],[1]],epochs=2)\\n"
              + "assert model2.predict([[0,0]]).shape==(1,1)\\n"
              + "model2.dispose()\\n"
            );
            await py.runPythonAsync(
                "import numpy as np\\n"
              + "from tensorflow.keras.models import Sequential\\n"
              + "from tensorflow.keras.layers import Dense\\n"
              + "import tensorflow as tf\\n"
              + "m3=Sequential([Dense(8,'relu'), Dense(4,'linear')])\\n"
              + "m3.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(0.03))\\n"
              + "xc=np.asarray([[0,0],[0,1],[1,0],[1,1]] * 12,dtype='float32')\\n"
              + "yc=np.asarray([0,1,2,3]*12,dtype='int32')\\n"
              + "hist3=await m3.fit(xc,yc,epochs=18,batch_size=12)\\n"
              + "assert 'loss' in hist3.history and len(hist3.history['loss'])==18\\n"
              + "assert np.isfinite(hist3.history['loss']).all()\\n"
              + "assert hist3.history['loss'][-1] < hist3.history['loss'][0]\\n"
              + "assert m3.predict(xc[:4]).shape == (4,4)\\n"
              + "assert m3.layers[0].get_weights()[0].shape == (2,8)\\n"
              + "m3.dispose()\\n"
            );
            await py.runPythonAsync(
                "import numpy as np\\n"
              + "import tensorflow as tf\\n"
              + "from tensorflow.keras.models import Sequential\\n"
              + "from tensorflow.keras.layers import Dense\\n"
              + "from tensorflow.keras.activations import relu,linear\\n"
              + "from tensorflow.keras.losses import SparseCategoricalCrossentropy\\n"
              + "from tensorflow.keras.optimizers import Adam\\n"
              + "from tensorflow.keras.regularizers import l2\\n"
              + "m4=Sequential([Dense(4,activation='relu',kernel_regularizer=l2(0.1),name='L1'),Dense(3,activation='linear',name='L2')])\\n"
              + "m4.compile(loss=SparseCategoricalCrossentropy(from_logits=True),optimizer=Adam(lr=0.01))\\n"
              + "m4.build((None,2))\\n"
              + "assert m4.input.shape.as_list()==[None,2]\\n"
              + "assert m4.layers[0].output.shape.as_list()==[None,4]\\n"
              + "assert m4.layers[0].activation is relu and m4.layers[1].activation is linear\\n"
              + "assert type(m4.layers[0].kernel_regularizer) is l2\\n"
              + "assert m4.loss.get_config()['from_logits'] and isinstance(m4.optimizer,Adam)\\n"
              + "assert np.isclose(m4.optimizer.learning_rate.numpy(),0.01)\\n"
              + "xs=np.asarray([[0,0],[1,0],[0,1],[1,1]]*2,dtype='float32')\\n"
              + "ys=np.asarray([0,1,2,1]*2,dtype='int32')\\n"
              + "h4=await m4.fit(xs,ys,epochs=2)\\n"
              + "assert np.isfinite(h4.history['loss']).all()\\n"
              + "assert (await m4.predict_async(xs)).shape==(8,3)\\n"
              + "assert len(await m4.get_weights_async())==4\\n"
              + "m4.dispose()\\n"
            );
            await py.runPythonAsync(
                "import tensorflow as tf\\n"
              + "import numpy as np\\n"
              + "assert tf.is_tensor(tf.constant([1.,2.]))\\n"
              + "x=tf.constant([[1.,2.],[3.,4.]])\\n"
              + "assert x.shape==(2,2) and x.dtype=='float32'\\n"
              + "np.testing.assert_allclose((x+1).numpy(),[[2,3],[4,5]])\\n"
              + "np.testing.assert_allclose(tf.matmul(x,x).numpy(),[[7,10],[15,22]])\\n"
              + "assert np.isclose(float(tf.reduce_sum(x)),10.)\\n"
              + "assert np.isclose(float(tf.reduce_mean(x)),2.5)\\n"
              + "np.testing.assert_allclose(tf.nn.softmax(tf.constant([[1.,2.]])).numpy().sum(axis=1),[1.])\\n"
              + "np.testing.assert_allclose((await x.numpy_async()),[[1,2],[3,4]])\\n"
              + "np.testing.assert_allclose(tf.reshape(x,[4]).numpy(),[1,2,3,4])\\n"
              + "np.testing.assert_allclose(tf.one_hot([0,2],3).numpy(),[[1,0,0],[0,0,1]])\\n"
              + "v=tf.Variable([2.,3.])\\n"
              + "v.assign_add([1.,1.])\\n"
              + "np.testing.assert_allclose(v.numpy(),[3,4])\\n"
              + "from browser_ml.tfjs_tensor import grad\\n"
              + "g=grad(lambda t: tf.reduce_sum(t*t))(tf.constant([1.,2.,3.]))\\n"
              + "np.testing.assert_allclose(g.numpy(),[2,4,6])\\n"
            );
            await py.runPythonAsync(
                "import numpy as np\\n"
              + "import tensorflow as tf\\n"
              + "from tensorflow.keras.models import Sequential\\n"
              + "from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization\\n"
              + "cnn=Sequential([Conv2D(2,3,activation='relu',input_shape=(8,8,1)),MaxPooling2D(2),Flatten(),Dropout(0.1),Dense(2,activation='softmax')])\\n"
              + "cnn.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(0.005))\\n"
              + "images=np.arange(8*8*8,dtype='float32').reshape((8,8,8,1))/512\\n"
              + "labels=np.eye(2,dtype='float32')[[0,1,0,1,0,1,0,1]]\\n"
              + "hist=await cnn.fit(images,labels,epochs=2,batch_size=4)\\n"
              + "assert len(hist.history['loss'])==2\\n"
              + "assert cnn.predict(images).shape==(8,2)\\n"
              + "assert len(cnn.layers[0].get_weights())==2\\n"
              + "cnn.dispose()\\n"
              + "bn=Sequential([Dense(4,input_shape=(2,)),BatchNormalization(),Dense(1)])\\n"
              + "bn.compile(loss=tf.keras.losses.MeanSquaredError())\\n"
              + "bh=await bn.fit(np.ones((8,2),dtype='float32'),np.ones((8,1),dtype='float32'),epochs=1)\\n"
              + "assert len(bh.history['loss'])==1\\n"
              + "bn.dispose()\\n"
            );
            self.postMessage({ok:true,backend:output});
          } catch (err) {self.postMessage({ok:false,error:String(err),stack:err.stack})}
        }`;
      return await new Promise((resolve,reject) => {
        const blob=URL.createObjectURL(new Blob([code],{type:'text/javascript'}));
        const worker=new Worker(blob,{type:'module'});
        const timer=setTimeout(()=>{worker.terminate();reject(Error('Pyodide/TF.js smoke test timed out'));},150000);
        worker.onmessage=e=>{clearTimeout(timer);worker.terminate();URL.revokeObjectURL(blob);resolve(e.data)};
        worker.onerror=e=>{clearTimeout(timer);worker.terminate();reject(Error(e.message||'Worker error'))};
        worker.postMessage({pyodide:`http://127.0.0.1:${port}/pyodide/pyodide.mjs`,pyodideBase:`http://127.0.0.1:${port}/pyodide/`,bridge:`http://127.0.0.1:${port}/vendor/tfjs_bridge.py`,compat:`http://127.0.0.1:${port}/vendor/tfjs_compat.py`,tensor:`http://127.0.0.1:${port}/vendor/tfjs_tensor.py`,layers:`http://127.0.0.1:${port}/vendor/tfjs_layers.py`,losses:`http://127.0.0.1:${port}/vendor/course-losses.js`,tfjs:src});
      });
    },port);
    console.log('Browser scientific/TensorFlow.js backend test:', JSON.stringify(outcome));
    assert.equal(outcome.ok,true,outcome.error+'\n'+outcome.stack);
  } finally {
    if (browser) await browser.close();
    server.kill();
  }
})().catch(e=>{console.error(e);process.exitCode=1});
