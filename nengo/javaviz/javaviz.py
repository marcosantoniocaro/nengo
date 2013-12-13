import socket
import rpyc
import struct

import nengo
import numpy as np

import thread
import time

class View:
    def __init__(self, model, udp_port=56789, client='localhost'):
        self.rpyc = rpyc.classic.connect(client)

        attempts = 0
        while attempts<100:
            try:
                vr = self.rpyc.modules.timeview.javaviz.ValueReceiver(udp_port+attempts)
                break
            except:
                attempts += 1
        vr.start()
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #UDP 
        self.socket_target = (client, udp_port+attempts)
        
        self.socket_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket_recv.bind(('localhost', udp_port+attempts+1))
        thread.start_new_thread(self.receiver, ())

        label = model.label
        if label is None: label='Nengo Visualizer 0x%x'%id(model)
        net = self.rpyc.modules.nef.Network(label)
        
        control_ensemble = None
        remote_objs = {}
        ignore_connections = set()
        inputs = []
        for obj in model.objs[:]:
            if isinstance(obj, nengo.Ensemble):
                if control_ensemble is None:
                    e = self.rpyc.modules.timeview.javaviz.ControlEnsemble(vr, id(obj), obj)
                    e.init_control('localhost', udp_port+attempts+1)
                    control_ensemble = e
                else:    
                    e = self.rpyc.modules.timeview.javaviz.Ensemble(vr, id(obj), obj)
                net.add(e)
                remote_objs[obj] = e
                
                
                with model:                      
                    def send(t, x, self=self, format='>Lf'+'f'*obj.dimensions, id=id(obj)):
                        msg = struct.pack(format, id, t, *x)
                        self.socket.sendto(msg, self.socket_target)
                    
                        
                    node = nengo.Node(send, dimensions=obj.dimensions)
                    c = nengo.Connection(obj, node, filter=None)
                    ignore_connections.add(c)
            elif isinstance(obj, nengo.Node):
                if obj.dimensions == 0:
                    output = obj.output
                    
                    if callable(output): output = output(np.zeros(obj.dimensions))
                    if isinstance(output, (int, float)):
                        output_dims = 1
                    else:
                        output_dims = len(output)
                    obj._output_dims = output_dims    
                    input = net.make_input(obj.label, tuple([0]*output_dims))
                    obj.output = OverrideFunction(obj.output, id(input))
                    print 'make override', id(input)
                    remote_objs[obj] = input
                    inputs.append(input)
                    
        for input in inputs:
            control_ensemble.register(id(input), input)
                    
        for c in model.connections:
            if c not in ignore_connections:
                if c.pre in remote_objs and c.post in remote_objs:
                    pre = remote_objs[c.pre]
                    post = remote_objs[c.post]
                    dims = c.pre.dimensions if isinstance(c.pre, nengo.Ensemble) else c.pre._output_dims
                    t = post.create_new_dummy_termination(dims)
                    net.connect(pre, t)
                else:
                    print 'cannot process connection from %s to %s'%(`c.pre`, `c.post`)
                
        view = net.view()  
        control_ensemble.set_view(view)
    
    def receiver(self):
        print 'waiting for msg'
        while True:
            msg = self.socket_recv.recv(4096)
            time = struct.unpack('>f', msg[:4])
            OverrideFunction.overrides['block_time'] = time
            for i in range((len(msg)-4)/12):
                id, index, value = struct.unpack('>LLf', msg[4+i*12:16+i*12])
                print '  changed', id, index, value
                OverrideFunction.overrides[id][index]=value
            
class OverrideFunction(object):
    overrides = {'block_time':0.0}
    def __init__(self, function, id):
        self.function = function
        self.id = id
        OverrideFunction.overrides[id] = {}
    def __call__(self, t):
        while OverrideFunction.overrides['block_time'] < t:
            time.sleep(0.01)
        if callable(self.function):
            value = np.array(self.function(t))
        else:
            value = np.array(self.function)
        for k,v in OverrideFunction.overrides.get(self.id, {}).items():
            value[k] = v
        return value