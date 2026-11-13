import hashlib,json,pathlib,tempfile,unittest
import controller,pilot
P=pathlib.Path(__file__).parent
class Batch(unittest.TestCase):
 def test_exact_scope_and_environment_command(self):
  rows=json.loads((P/'CLAIM.json').read_text())['rows'];identity=json.loads((P/'IDENTITY.json').read_text())
  self.assertEqual(identity['original39_indices'],[2,3,4,13]);self.assertEqual(len(rows),4);self.assertEqual({r['op'] for r in rows},{'RoundFwdOp'})
  self.assertEqual([r['case_id'] for r in rows],['16M-float16','16M-float32','256M-bfloat16','256M-float16'])
  self.assertEqual(rows,identity['rows']);self.assertEqual(len({r['contract_sha256'] for r in rows}),4)
  for i in range(4):
   cmd=pilot.command(pathlib.Path('/root'),P,i,pathlib.Path('/out'))
   self.assertEqual(cmd[0],'/root/round-isolated-envs/round-candidate03/bin/python3')
   self.assertEqual(cmd[cmd.index('--index')+1],str(i));self.assertEqual(cmd[cmd.index('--windows')+1],'15')
   self.assertEqual(pathlib.Path(cmd[1]).name,'environment_entry.py')
  for i in [-1,4]:
   with self.assertRaises(ValueError):pilot.command(pathlib.Path('/root'),P,i,pathlib.Path('/out'))
 def test_serial(self):
  seen=[];rows,rc=controller.run_queue(lambda i,a,b:i,[(i,i) for i in range(4)],lambda i:seen.append(i) or 0,lambda a,b:'pass');self.assertEqual(rc,0);self.assertEqual(seen,[0,1,2,3])
 def test_each_failure_stops(self):
  for stop in range(4):
   for failure in [1,75,76,78,124,143]:
    seen=[];rows,rc=controller.run_queue(lambda i,a,b:i,[(i,i) for i in range(4)],lambda i:seen.append(i) or (failure if i==stop else 0),lambda a,b:'pass')
    self.assertEqual(rc,failure);self.assertEqual(seen,list(range(stop+1)))
 def test_each_gate_failure_stops(self):
  for stop in range(4):
   seen=[]
   def gate(a,b):
    if a==stop:raise ValueError('failed')
    return 'pass'
   rows,rc=controller.run_queue(lambda i,a,b:i,[(i,i) for i in range(4)],lambda i:seen.append(i) or 0,gate)
   self.assertEqual(rc,78);self.assertEqual(seen,list(range(stop+1)))
 def test_missing_environment_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   p=pathlib.Path(td)
   with self.assertRaises(FileNotFoundError):controller.gate(p,p)
   (p/'environment.json').write_text(json.dumps({'status':'failed'}))
   with self.assertRaisesRegex(ValueError,'environment'):controller.gate(p,p)
 def test_unchanged_runner(self):self.assertEqual(hashlib.sha256((P/'measure_driver_provenance_v2.py').read_bytes()).hexdigest(),pilot.RUNNER_SHA)
if __name__=='__main__':unittest.main(verbosity=2)
