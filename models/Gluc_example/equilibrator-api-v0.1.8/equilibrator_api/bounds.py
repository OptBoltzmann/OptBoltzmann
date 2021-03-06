import pkg_resources
import logging
import csv
from numpy import ones, log

from copy import deepcopy
from equilibrator_api.concs import ConcentrationConverter
from equilibrator_api import settings

class InvalidBounds(Exception):
    pass


class BaseBounds(object):
    """A base class for declaring bounds on things."""

    def Copy(self):
        """Returns a (deep) copy of self."""
        raise NotImplementedError

    def GetRange(self):
        """Returns a 2-tuple of the concentration range."""
        return None

    def GetLowerBound(self, key):
        """Get the lower bound for this key.

        Args:
            key: a string key.
        """
        raise NotImplementedError

    def GetUpperBound(self, key):
        """Get the upper bound for this key.

        Args:
            key: a string key.
        """
        raise NotImplementedError

    def GetBoundTuple(self, key):
        """Returns the bounds as a 2 tuple (lower, upper)."""
        return self.GetLowerBound(key), self.GetUpperBound(key)

    def GetBounds(self, keys):
        """Get the bounds for a set of keys in order.

        Args:
            keys: an iterable of keys.

        Returns:
            A two-tuple (lower_bounds, upper_bounds) where both
            items are Numpy arrays of dimensions 1xlen(keys)
        """
        lbs = ones((len(keys), 1))
        ubs = ones((len(keys), 1))
        for i, key in enumerate(keys):
            lbs[i, 0] = self.GetLowerBound(key)
            ubs[i, 0] = self.GetUpperBound(key)
        return lbs, ubs

    def GetLnBounds(self, keys):
        """Get the bounds for a set of keys in order.

        Args:
            keys: an iterable of keys.

        Returns:
            A two-tuple (lower_bounds, upper_bounds) where both
            items are Numpy arrays of dimensions 1xlen(keys)
        """
        lb, ub = self.GetBounds(keys)
        return log(lb), log(ub)

    def SetBounds(self, key, lb, ub):
        """Set bounds for a specific key.

        Args:
            key: the key for the bounds.
            lb: the lower bound value.
            ub: the upper bound value.
        """
        assert lb <= ub
        self.lower_bounds[key] = lb
        self.upper_bounds[key] = ub


class Bounds(BaseBounds):
    """Contains upper and lower bounds for various keys. Allows for defaults."""

    def __init__(self,
                 lower_bounds=None,
                 upper_bounds=None,
                 default_lb=None,
                 default_ub=None):
        """Initialize the bounds object.

        Args:
            lower_bounds: a dictionary mapping strings to float lower bounds.
            upper_bounds: a dictionary mapping strings to float upper bounds.
            default_lb: the default lower bound to return.
            default_lb: the default upper bound to return.
        """
        self.lower_bounds = lower_bounds or {}
        self.upper_bounds = upper_bounds or {}
        self.default_lb = default_lb
        self.default_ub = default_ub

        self.c_range = (self.default_lb, self.default_ub)

    @classmethod
    def from_csv_file(cls, f, default_lb=None, default_ub=None):
        lbs = {}
        ubs = {}
        for row in csv.DictReader(f):
            cid = row['cid']
            lb, ub = row.get('c_min'), row.get('c_max')
            if lb.strip():
                lb = float(lb)
            else:
                lb = None
            if ub.strip():
                ub = float(ub)
            else:
                ub = None

            lbs[cid] = lb
            ubs[cid] = ub

        bounds = Bounds(lbs, ubs, default_lb, default_ub)
        bounds._check_bounds()
        return bounds

    @classmethod
    def from_csv_filename(cls, fname,
                          default_lb=settings.DEFAULT_CONC_LB,
                          default_ub=settings.DEFAULT_CONC_UB):
        with open(fname, 'rU') as f:
            return cls.from_csv_file(f, default_lb=default_lb, default_ub=default_ub)

    @classmethod
    def from_dataframe(cls, df, unit='Molar',
                       default_lb=settings.DEFAULT_CONC_LB,
                       default_ub=settings.DEFAULT_CONC_UB):
        lbs = {}
        ubs = {}

        for idx in df.index:
            row = df.loc[idx]
            cid = row['Compound:Identifiers:kegg.compound']
            ub = float(row['Concentration:Max'])
            lb = float(row['Concentration:Min'])
            ub = ConcentrationConverter.to_molar_string(ub, unit)
            lb = ConcentrationConverter.to_molar_string(lb, unit)

            ubs[cid] = ub
            lbs[cid] = lb

        bounds = Bounds(lbs, ubs, default_lb, default_ub)
        bounds._check_bounds()
        return bounds

    def _check_bounds(self):
        if self.default_lb > self.default_ub:
            msg = (
                'Default lower bound %.2g > default upper bound %.2g' %
                (self.default_lb, self.default_ub))
            raise InvalidBounds(msg)

        for cid in self.upper_bounds:
            lb = self.GetLowerBound(cid)
            ub = self.GetUpperBound(cid)
            if lb > ub:
                msg = (
                    'Invalid bounds for %s: lower bound %f > upper bound %f' %
                    (cid, lb, ub))
                raise InvalidBounds(msg)

    def Copy(self):
        """Returns a deep copy of self."""
        new_lb = deepcopy(self.lower_bounds)
        new_ub = deepcopy(self.upper_bounds)
        return Bounds(new_lb, new_ub,
                      self.default_lb,
                      self.default_ub)

    def GetRange(self):
        """Returns a 2-tuple of the concentration range."""
        return self.c_range

    def GetLowerBound(self, key):
        """Get the lower bound for this key.

        Args:
            key: a string key.
        """
        val = self.lower_bounds.get(key) or self.default_lb
        return val

    def GetUpperBound(self, key):
        """Get the upper bound for this key.

        Args:
            key: a string key.
        """
        val = self.upper_bounds.get(key) or self.default_ub
        return val

COFACTORS_FNAME = pkg_resources.resource_filename(
    'equilibrator_api', 'data/cofactors.csv')
DEFAULT_BOUNDS = Bounds.from_csv_filename(COFACTORS_FNAME)
